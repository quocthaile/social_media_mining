---
license: mit
task_categories:
- text-classification
language:
- vi
pretty_name: ViHSD
size_categories:
- 10K<n<100K
---
# Dataset Card for Dataset Name

ViHSD - Vietnamese Hate Speech Detection Dataset 

## Dataset Details

### Dataset Description

The dataset contains about 33K annotated comments from social networks. Each has one of three labels: HATE, OFFENSIVE, CLEAN 


## Uses

Use directly from the Hugging face dataset loader

### Direct Use

```
from datasets import load_dataset

train = load_dataset("sonlam1102/vihsd", split="train")
dev = load_dataset("sonlam1102/vihsd", split="validation")
test = load_dataset("sonlam1102/vihsd", split="test")
```

[More Information Needed]

## Citation

**BibTeX:**

```
@inproceedings{luu2021large,
  title={A large-scale dataset for hate speech detection on vietnamese social media texts},
  author={Luu, Son T and Nguyen, Kiet Van and Nguyen, Ngan Luu-Thuy},
  booktitle={Advances and Trends in Artificial Intelligence. Artificial Intelligence Practices: 34th International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems, IEA/AIE 2021, Kuala Lumpur, Malaysia, July 26--29, 2021, Proceedings, Part I 34},
  pages={415--426},
  year={2021},
  organization={Springer}
}
```

## Dataset Contact 

Email: sonlt@uit.edu.vn (Alternative: son.lt1103@gmail.com)