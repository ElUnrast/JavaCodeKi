import os
import random
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    PreTrainedTokenizer,
    T5TokenizerFast as T5Tokenizer
)
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
from pandas import read_csv
from base64 import b64decode


class SourceCodeDataSet:
    def __init__(self, filename):
        # path;methodSignature;changePositions;checkinComment;
        # methodContentOld; methodContentNew;methodContentCanonicOld;methodContentCanonicNew
        self.data_frame = read_csv(filename, delimiter=';', header=0, keep_default_na=False)
        self.data_frame['methodContentOld'] = self.data_frame['methodContentOld'].apply(lambda s: b64decode(s).decode('utf-8'))
        self.data_frame['methodContentNew'] = self.data_frame['methodContentNew'].apply(lambda s: b64decode(s).decode('utf-8'))
        self.data_frame['checkinComment'] = self.data_frame['checkinComment'].apply(lambda s: b64decode(s).decode('utf-8'))
        self.data_frame['changePositions'] = self.data_frame['changePositions'].apply(self._read_all_change_positions)

    def _read_change_position(self, change):
        splitted = change.split(',')
        return {'line': int(splitted[0]), 'size1': int(splitted[1]), 'size2': int(splitted[2])}

    def _read_all_change_positions(self, text):
        return [self._read_change_position(change) for change in text[1:-1].split(')(')]


class SourceCodeKi:
    def __init__(self, model_name='flax-community/git-t5-base', git_repository=None, max_token_length=512):
        self.git_repository = git_repository
        self.trained_model_dir = model_name
        self.max_token_length = max_token_length
        self.tasks = ['patch', 'fix', 'mname', 'vname', 'complete']

        if torch.cuda.is_available():
            self.use_gpu = True
            self.device = torch.device("cuda")
        else:
            self.use_gpu = False
            self.device = torch.device("cpu")

        if os.path.isfile(f'{self.trained_model_dir}/pytorch_model.bin'):
            print(f'Loading local Model: {self.trained_model_dir}')
            self.tokenizer, self.model = T5Tokenizer.from_pretrained(f"{model_name}"), T5ForConditionalGeneration.from_pretrained(f"{model_name}", return_dict=True)
        else:
            model_name = 'flax-community/git-t5-base'
            self.tokenizer, self.model = T5Tokenizer.from_pretrained(f"{model_name}"), T5ForConditionalGeneration.from_pretrained(f"{model_name}", return_dict=True)

        self.model.to(self.device)
        self.T5Model = LightningModel(tokenizer=self.tokenizer, model=self.model, outputdir=self.trained_model_dir)

    def predict(self, text):
        return self.model.predict(text)

    def do_text_manipulation(self, source_text):
        broken_text = source_text

        for _ in range(int(len(broken_text.split(" "))/4)):
            if(random.randint(0, 100) > 30):
                randc = random.choice(broken_text.split(" "))
                broken_text = broken_text.replace(randc, ''.join(random.choice('abcdefghijklmnopqrstuvxyz()}{!=/+-*1234567890%') for _ in range(len(randc))).lower())

        return broken_text

    def mask_methodname(self, method_code: str):
        end_index = method_code.index('(')
        begin_index = method_code.rfind(' ', 0, end_index)
        return method_code[:begin_index] + ' method' + method_code[end_index:]

    def train(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        batch_size: int = 4,
        max_epochs: int = 5,
        early_stopping_patience_epochs: int = 3,  # 0 to disable early stopping feature
        precision=32
    ):
        """
        trains T5/MT5 model on custom dataset

        Args:
            train_df (pd.DataFrame): training datarame. Dataframe must have 2 column --> "source_text" and "target_text"
            eval_df ([type], optional): validation datarame. Dataframe must have 2 column --> "source_text" and "target_text"
            batch_size (int, optional): batch size. Defaults to 8.
            max_epochs (int, optional): max number of epochs. Defaults to 5.
            early_stopping_patience_epochs (int, optional): monitors val_loss on epoch end and stops training, if val_loss does not improve after the specied number of epochs. set 0 to disable early stopping. Defaults to 0 (disabled)
            precision (int, optional): sets precision training - Double precision (64), full precision (32) or half precision (16). Defaults to 32.
        """
        data_module = LightningDataModule(
            train_df,
            eval_df,
            self.tokenizer,
            batch_size=batch_size,
            source_max_token_len=self.max_token_length,
            target_max_token_len=self.max_token_length
        )

        early_stop_callback = (
            [
                EarlyStopping(
                    monitor="val_loss",
                    min_delta=0.00,
                    patience=early_stopping_patience_epochs,
                    verbose=True,
                    mode="min",
                )
            ]
            if early_stopping_patience_epochs > 0
            else None
        )

        gpus = 1 if self.use_gpu else 0

        trainer = pl.Trainer(
            callbacks=early_stop_callback,
            max_epochs=max_epochs,
            gpus=gpus,
            progress_bar_refresh_rate=5,
            precision=precision
        )

        trainer.fit(self.T5Model, data_module)
        trainer.optimizers
        return trainer

    def prepare_data(self, task: str, ds_filename: str):
        source_dataset = SourceCodeDataSet(ds_filename)
        pandas_df = source_dataset.data_frame
        # path;methodSignature;changePositions;checkinComment;
        # methodContentOld; methodContentNew;methodContentCanonicOld;methodContentCanonicNew
        df = pandas_df.drop(['path', 'changePositions'], axis=1)

        if task == 'patch':
            df = pandas_df.drop(['methodSignature', 'checkinComment', 'methodContentCanonicOld', 'methodContentCanonicNew'], axis=1)
            df = df.rename(columns={'methodContentOld': 'source_text', 'methodContentNew': 'target_text'})
        elif task == 'fix':
            df = pandas_df.drop(['methodSignature', 'checkinComment', 'methodContentCanonicOld', 'methodContentCanonicNew', 'methodContentOld'], axis=1)
            df = df.rename(columns={'methodContentNew': 'target_text'})
            df['source_text'] = df['target_text']
            df['source_text'] = df['source_text'].apply(lambda x: self.do_text_manipulation(x))
        elif task == 'mname':
            df = pandas_df.drop(['checkinComment', 'methodContentCanonicOld', 'methodContentCanonicNew', 'methodContentOld'], axis=1)
            df = df.rename(columns={'methodContentNew': 'source_text', 'methodSignature': 'target_text'})
            df['target_text'] = df['source_text'].apply(lambda x: x[0:x.index('(')])
            df['source_text'] = df['source_text'].apply(lambda x: self.mask_methodname(x))
        elif task == 'vname':
            df = pandas_df.drop(['methodSignature', 'checkinComment', 'methodContentOld', 'methodContentCanonicOld'], axis=1)
            df = df.rename(columns={'methodContentCanonicNew': 'source_text', 'methodContentNew': 'target_text'})
        elif task == 'complete':
            df = pandas_df.drop(['methodSignature', 'checkinComment', 'methodContentOld', 'methodContentCanonicOld', 'methodContentCanonicNew'], axis=1)
            df = df.rename(columns={'methodContentNew': 'source_text', 'methodContentNew': 'target_text'})
            df['target_text'] = df['source_text'].apply(lambda x: x[len(x)//2:])
            df['source_text'] = df['source_text'].apply(lambda x: x[0:len(x)//2])
        else:
            raise ValueError()

        df['source_text'] = df['source_text'].apply(lambda s: f'{task}: {s}')

        source_max_token_len = self.max_token_length
        target_max_token_len = self.max_token_length

        df['source_text_tokenizer_result'] = df['source_text'].apply(lambda x: self.tokenizer(
            x, add_special_tokens=True, padding="max_length", max_length=source_max_token_len, return_tensors="pt"
        ))
        df['target_text_tokenizer_result'] = df['target_text'].apply(lambda x: self.tokenizer(
            x, add_special_tokens=True, padding="max_length", max_length=target_max_token_len, return_tensors="pt"
        ))
        df['source_valid'] = df['source_text_tokenizer_result'].apply(lambda x: len(x.input_ids.squeeze()) <= source_max_token_len)
        df['target_valid'] = df['target_text_tokenizer_result'].apply(lambda x: len(x.input_ids.squeeze()) <= target_max_token_len)
        df = df[df['source_valid'] & df['target_valid']]
        df_valid = df.drop(['source_text_tokenizer_result', 'target_text_tokenizer_result', 'source_valid', 'target_valid'], axis=1)
        print(f'{len(df_valid)} examples with valid length {source_max_token_len}/{target_max_token_len} out of {len(df)} examples found')

        train, test = train_test_split(df_valid, test_size=0.2)
        return train, test

    def train_data(self, train, test):
        for i in range(0, len(train) - 100, 100):
            print(f'range start: {i:d}')
            train = train.iloc[i:i + 100]
            test = test.iloc[i // 5:(i // 5) + 20]
            trainer = self.train(train_df=train, eval_df=test, batch_size=4)
            trainer.save_model()

            if self.use_gpu:
                self.wipe_out_memory(trainer)

            del trainer
            torch.cuda.empty_cache()

    def wipe_out_memory(self, trainer):
        device = 'cpu'
        for k, o in trainer._lightning_optimizers.items():
            o._trainer = None
            for param in o._optimizer.state.values():
                # Not sure there are any global tensors in the state dict
                if isinstance(param, torch.Tensor):
                    param.data = param.data.to(device)
                    if param._grad is not None:
                        param._grad.data = param._grad.data.to(device)
                elif isinstance(param, dict):
                    for subparam in param.values():
                        if isinstance(subparam, torch.Tensor):
                            subparam.data = subparam.data.to(device)
                            if subparam._grad is not None:
                                subparam._grad.data = subparam._grad.data.to(device)
            del o

    def predict(
        self,
        source_text: str,
        max_length: int = 512,
        num_return_sequences: int = 1,
        num_beams: int = 2,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 2.5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ):
        """
        generates prediction for T5/MT5 model

        Args:
            source_text (str): any text for generating predictions
            max_length (int, optional): max token length of prediction. Defaults to 512.
            num_return_sequences (int, optional): number of predictions to be returned. Defaults to 1.
            num_beams (int, optional): number of beams. Defaults to 2.
            top_k (int, optional): Defaults to 50.
            top_p (float, optional): Defaults to 0.95.
            do_sample (bool, optional): Defaults to True.
            repetition_penalty (float, optional): Defaults to 2.5.
            length_penalty (float, optional): Defaults to 1.0.
            early_stopping (bool, optional): Defaults to True.
            skip_special_tokens (bool, optional): Defaults to True.
            clean_up_tokenization_spaces (bool, optional): Defaults to True.

        Returns:
            list[str]: returns predictions
        """
        input_ids = self.tokenizer.encode(
            source_text, return_tensors="pt", add_special_tokens=True
        )
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            generated_ids = self.model.generate(
                input_ids=input_ids,
                num_beams=num_beams,
                max_length=max_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
            )
            preds = [
                self.tokenizer.decode(
                    g,
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )
                for g in generated_ids
            ]
            return preds


class PyTorchDataModule(Dataset):
    """  PyTorch Dataset class  """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
    ):
        """
        initiates a PyTorch Dataset Module for input data

        Args:
            data (pd.DataFrame): input pandas dataframe. Dataframe must have 2 column --> "source_text" and "target_text"
            tokenizer (PreTrainedTokenizer): a PreTrainedTokenizer (T5Tokenizer, MT5Tokenizer, or ByT5Tokenizer)
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
        """
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = data['source_text'].str.len().max()
        self.target_max_token_len = data['target_text'].str.len().max()

    def __len__(self):
        """ returns length of data """
        return len(self.data)

    def __getitem__(self, index: int):
        """ returns dictionary of input tensors to feed into T5/MT5 model"""

        data_row = self.data.iloc[index]
        source_text = data_row["source_text"]
        target_text = data_row["target_text"]

        source_text_encoding = self.tokenizer(
            source_text,
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        target_text_encoding = self.tokenizer(
            target_text,
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        labels = target_text_encoding["input_ids"]
        labels[
            labels == 0
        ] = -100  # to make sure we have correct labels for T5 text generation

        return dict(
            source_text=source_text,
            target_text=target_text,
            source_text_input_ids=source_text_encoding["input_ids"].flatten(),
            source_text_attention_mask=source_text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=target_text_encoding["attention_mask"].flatten(),
        )


class LightningDataModule(pl.LightningDataModule):
    """ PyTorch Lightning data class """

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
    ):
        """
        initiates a PyTorch Lightning Data Module

        Args:
            train_df (pd.DataFrame): training dataframe. Dataframe must contain 2 columns --> "source_text" & "target_text"
            test_df (pd.DataFrame): validation dataframe. Dataframe must contain 2 columns --> "source_text" & "target_text"
            tokenizer (PreTrainedTokenizer): PreTrainedTokenizer (T5Tokenizer, MT5Tokenizer, or ByT5Tokenizer)
            batch_size (int, optional): batch size. Defaults to 4.
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
        """
        super().__init__()

        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def setup(self, stage=None):
        self.train_dataset = PyTorchDataModule(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )
        self.test_dataset = PyTorchDataModule(
            self.test_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )

    def train_dataloader(self):
        """ training dataloader """
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

    def test_dataloader(self):
        """ test dataloader """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

    def val_dataloader(self):
        """ validation dataloader """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )


class LightningModel(pl.LightningModule):
    """ PyTorch Lightning Model class"""

    def __init__(self, tokenizer, model, outputdir: str = "outputs"):
        """
        initiates a PyTorch Lightning Model

        Args:
            tokenizer : T5/MT5/ByT5 tokenizer
            model : T5/MT5/ByT5 model
            outputdir (str, optional): output directory to save model checkpoints. Defaults to "outputs".
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.outputdir = outputdir

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        """ forward step """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_size):
        """ training step """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_size):
        """ validation step """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_size):
        """ test step """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """ configure optimizers """
        return AdamW(self.parameters(), lr=0.0001)

    def training_epoch_end(self, training_step_outputs):
        """ save tokenizer and model on epoch end """
        avg_traning_loss = np.round(
            torch.mean(torch.stack([x["loss"] for x in training_step_outputs])).item(),
            4,
        )
        # path = f"{self.outputdir}/SimpleT5-epoch-{self.current_epoch}-train-loss-{str(avg_traning_loss)}"
        path = self.outputdir
        # sollte sich nicht verändert haben
        self.tokenizer.save_pretrained(path)
        print(f'Saving Model to: {path}')
        self.model.save_pretrained(path)
