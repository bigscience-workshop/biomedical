from abc import ABCMeta
import pandas as pd
import itertools
import numpy as np


class DatasetPrompts(metaclass=ABCMeta):

    def _init_dataset(self):
        raise NotImplementedError

    def _init_prompts(self):
        raise NotImplementedError

    def add_prompt(self,
                   func,
                   name,
                   answer_keys=None,
                   original_task=False,
                   answers_in_prompt=False,
                   metrics=None):

        self._prompts[name] = func
        self._metadata[name] = {
            'answer_keys': answer_keys,
            'original_task': original_task,
            'answers_in_prompt': answers_in_prompt,
            'metrics': metrics
        }

    def get_prompts(self):
        """
        Create a pandas dataframe for prompts
        :return:
        """
        data = []
        for split,dataset in self.splits.items():
            for name in self._prompts:
                # create prompted instances
                f = self._prompts[name]
                prompts = [f(x) for x in dataset]
                # multiple prompts per instance
                if len([True for x in prompts if type(x) is list]) > 1:
                    prompts = list(itertools.chain.from_iterable(prompts))

                names = np.array([split, name] * len(prompts)).reshape(-1,2)
                prompts = np.array(prompts).reshape(-1,1)
                prompts = np.hstack((names, prompts))
                print(prompts.shape)
                data.append(prompts)

                # generate metadata
                # TODO: dump to file
                # m = self._metadata[name]
                # row = [
                #     split,
                #     name,
                #     '|||'.join(m['answer_keys']) if m['answer_keys'] else m['answer_keys'],
                #     m['original_task'],
                #     m['answers_in_prompt'],
                #     '|||'.join(m['metrics']) if m['metrics'] else m['metrics']
                # ]

        return pd.DataFrame(data=np.vstack(data),
                            columns=['split', 'prompt_name', 'prompted_x'])