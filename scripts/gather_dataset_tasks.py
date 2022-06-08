#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate counts of tasks and fine-grained taks
"""

from bigbio.dataloader import BigBioConfigHelpers


def main():
    """
    Gather counts on tasks and fine-grained tasks
    """

    configs = BigBioConfigHelpers()

    dataset_task = set()

    for conf in configs:
        for task in conf.tasks:
            dataset_task.add(conf.dataset_name, str(task))

    print(dataset_task)


if __name__ == "__main__":
    main()
