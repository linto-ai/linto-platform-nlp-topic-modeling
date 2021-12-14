# linto-platform-nlp-topic-modeling

## Description
This repository is for building a Docker image for LinTO's NLP service: Topic Modeling on the basis of [linto-platform-nlp-core](https://github.com/linto-ai/linto-platform-nlp-core), can be deployed along with [LinTO stack](https://github.com/linto-ai/linto-platform-stack) or in a standalone way (see Develop section in below).

linto-platform-nlp-topic-modeling is backed by [spaCy](https://spacy.io/) v3.0+ featuring transformer-based pipelines, thus deploying with GPU support is highly recommeded for inference efficiency.

LinTo's NLP services adopt the basic design concept of spaCy: [component and pipeline](https://spacy.io/usage/processing-pipelines), components are decoupled from the service and can be easily re-used in other projects, components are organised into pipelines for realising specific NLP tasks. 

This service uses [FastAPI](https://fastapi.tiangolo.com/) to serve custom spaCy's components as pipelines:
- `topic`: Topic Modeling

## Usage

See documentation : [https://doc.linto.ai](https://doc.linto.ai)

## Deploy

With our proposed stack [https://github.com/linto-ai/linto-platform-stack](https://github.com/linto-ai/linto-platform-stack)

# Develop

## Build and run
1 Download models into `./assets` on the host machine (can be stored in other places), make sure that `git-lfs`: [Git Large File Storage](https://git-lfs.github.com/) is installed and availble at `/usr/local/bin/git-lfs`.
```bash
cd linto-platform-nlp-topic-modeling/
bash scripts/download_models.sh
```

2 configure running environment variables
```bash
mv .envdefault .env
# cat .envdefault
# APP_LANG=fr en | Running language of application, "fr en", "fr", etc.
# ASSETS_PATH_ON_HOST=./assets | Storage path of models on host. (only applicable when docker-compose is used)
# ASSETS_PATH_IN_CONTAINER=/app/assets | Volume mount point of models in container. (only applicable when docker-compose is used)
# WORKER_NUMBER=1 | Number of processing workers. (only applicable when docker-compose is used)
```

4 Build image
```bash
sudo docker build --tag lintoai/linto-platform-nlp-topic-modeling:latest .
```
or
```bash
sudo docker-compose build
```

5 Run container with GPU support, make sure that [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian) and GPU driver are installed.
```bash
sudo docker run --gpus all \
--rm -p 80:80 \
-v $PWD/assets:/app/assets:ro \
--env-file .env \
lintoai/linto-platform-nlp-topic-modeling:latest \
--workers 1
```
or
```bash
sudo docker-compose up
```
<details>
  <summary>Check running with CPU only setting</summary>
  
  - remove `--gpus all` from the first command.
  - remove `runtime: nvidia` from the `docker-compose.yml` file.
</details>


6 Navigate to `http://localhost/docs` or `http://localhost/redoc` in your browser, to explore the REST API interactively. See the examples for how to query the API.


## Specification for `http://localhost/topic/{lang}`

### Supported languages
| {lang} | Model | Size |
| --- | --- | --- |
| `en` | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 80 MB |
| `fr` | [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | 418 MB |

### Request
Please use `" | "` (with a white-space on the left and right side) to seperate the segments (e.g., sentences, paragraphs, documents, etc.), which will be considered as the units for topic modeling.

The example in below of two topics consisting the first paragraphs about GAFAM and Supervised/Unsupervised/Semi-supervised/Reinforcement/Deep Learning, extracted from Wikipedia.
```json
{
  "articles": [
    {
      "text": "Google LLC is an American multinational technology company that specializes in Internet-related services and products, which include online advertising technologies, a search engine, cloud computing, software, and hardware. It is considered one of the Big Five companies in the American information technology industry, along with Amazon, Apple, Meta (Facebook) and Microsoft. | Amazon.com, Inc. is an American multinational technology company which focuses on e-commerce, cloud computing, digital streaming, and artificial intelligence. It is one of the Big Five companies in the U.S. information technology industry, along with Google (Alphabet), Apple, Meta (Facebook), and Microsoft. The company has been referred to as one of the most influential economic and cultural forces in the world, as well as the world's most valuable brand. | Meta Platforms, Inc., doing business as Meta and formerly known as Facebook, Inc., is a multinational technology conglomerate based in Menlo Park, California. The company is the parent organization of Facebook, Instagram, and WhatsApp, among other subsidiaries. Meta is one of the world's most valuable companies and is considered one of the Big Tech companies in U.S. information technology, alongside Amazon, Google, Apple, and Microsoft. The company generates a substantial share of its revenue from the sale of advertisement placements to marketers. | Apple Inc. is an American multinational technology company that specializes in consumer electronics, computer software and online services. Apple is the largest information technology company by revenue (totaling $274.5 billion in 2020) and, since January 2021, the world's most valuable company. As of 2021, Apple is the fourth-largest PC vendor by unit sales and fourth-largest smartphone manufacturer. It is one of the Big Five American information technology companies, alongside Amazon, Google (Alphabet), Facebook (Meta), and Microsoft. | Microsoft Corporation is an American multinational technology corporation which produces computer software, consumer electronics, personal computers, and related services. Its best-known software products are the Microsoft Windows line of operating systems, the Microsoft Office suite, and the Internet Explorer and Edge web browsers. Its flagship hardware products are the Xbox video game consoles and the Microsoft Surface lineup of touchscreen personal computers. Microsoft ranked No. 21 in the 2020 Fortune 500 rankings of the largest United States corporations by total revenue; it was the world's largest software maker by revenue as of 2016. It is considered one of the Big Five companies in the U.S. information technology industry, along with Amazon, Google (Alphabet), Apple, and Facebook (Meta). | Supervised learning (SL) is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples. In supervised learning, each example is a pair consisting of an input object (typically a vector) and a desired output value (also called the supervisory signal). A supervised learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples. An optimal scenario will allow for the algorithm to correctly determine the class labels for unseen instances. This requires the learning algorithm to generalize from the training data to unseen situations in a reasonable way (see inductive bias). This statistical quality of an algorithm is measured through the so-called generalization error. | Unsupervised learning is a type of machine learning in which the algorithm is not provided with any pre-assigned labels or scores for the training data.[1][2] As a result, unsupervised learning algorithms must first self-discover any naturally occurring patterns in that training data set. Common examples include clustering, where the algorithm automatically groups its training examples into categories with similar features, and principal component analysis, where the algorithm finds ways to compress the training data set by identifying which features are most useful for discriminating between different training examples, and discarding the rest. This contrasts with supervised learning in which the training data include pre-assigned category labels (often by a human, or from the output of non-learning classification algorithm). Other intermediate levels in the supervision spectrum include reinforcement learning, where only numerical scores are available for each training example instead of detailed tags, and semi-supervised learning where only a portion of the training data have been tagged. | Semi-supervised learning is an approach to machine learning that combines a small amount of labeled data with a large amount of unlabeled data during training. Semi-supervised learning falls between unsupervised learning (with no labeled training data) and supervised learning (with only labeled training data). It is a special instance of weak supervision. | Reinforcement learning (RL) is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning. | Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised."
    }
  ]
}
```

### Response
In the response results, sometimes a topic with topic_id `-1` is presented, which refers to noise topic and correponds to outlier input segments, can typically be ignored.

`"count"` refers to the topic frequency (number of segments attached to the topic), `"phrases"` represents a list of representative phrases of the topic with associated c-TF-IDF scores.

`"topic_assignments"` shows the list of segments, their assignments to a specific topic, and probabilities over all topics.
```json
{
  "topic": [
    {
      "text": "Google LLC is an American multinational technology company that specializes in Internet-related services and products, which include online advertising technologies, a search engine, cloud computing, software, and hardware. It is considered one of the Big Five companies in the American information technology industry, along with Amazon, Apple, Meta (Facebook) and Microsoft. | Amazon.com, Inc. is an American multinational technology company which focuses on e-commerce, cloud computing, digital streaming, and artificial intelligence. It is one of the Big Five companies in the U.S. information technology industry, along with Google (Alphabet), Apple, Meta (Facebook), and Microsoft. The company has been referred to as one of the most influential economic and cultural forces in the world, as well as the world's most valuable brand. | Meta Platforms, Inc., doing business as Meta and formerly known as Facebook, Inc., is a multinational technology conglomerate based in Menlo Park, California. The company is the parent organization of Facebook, Instagram, and WhatsApp, among other subsidiaries. Meta is one of the world's most valuable companies and is considered one of the Big Tech companies in U.S. information technology, alongside Amazon, Google, Apple, and Microsoft. The company generates a substantial share of its revenue from the sale of advertisement placements to marketers. | Apple Inc. is an American multinational technology company that specializes in consumer electronics, computer software and online services. Apple is the largest information technology company by revenue (totaling $274.5 billion in 2020) and, since January 2021, the world's most valuable company. As of 2021, Apple is the fourth-largest PC vendor by unit sales and fourth-largest smartphone manufacturer. It is one of the Big Five American information technology companies, alongside Amazon, Google (Alphabet), Facebook (Meta), and Microsoft. | Microsoft Corporation is an American multinational technology corporation which produces computer software, consumer electronics, personal computers, and related services. Its best-known software products are the Microsoft Windows line of operating systems, the Microsoft Office suite, and the Internet Explorer and Edge web browsers. Its flagship hardware products are the Xbox video game consoles and the Microsoft Surface lineup of touchscreen personal computers. Microsoft ranked No. 21 in the 2020 Fortune 500 rankings of the largest United States corporations by total revenue; it was the world's largest software maker by revenue as of 2016. It is considered one of the Big Five companies in the U.S. information technology industry, along with Amazon, Google (Alphabet), Apple, and Facebook (Meta). | Supervised learning (SL) is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples. In supervised learning, each example is a pair consisting of an input object (typically a vector) and a desired output value (also called the supervisory signal). A supervised learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples. An optimal scenario will allow for the algorithm to correctly determine the class labels for unseen instances. This requires the learning algorithm to generalize from the training data to unseen situations in a reasonable way (see inductive bias). This statistical quality of an algorithm is measured through the so-called generalization error. | Unsupervised learning is a type of machine learning in which the algorithm is not provided with any pre-assigned labels or scores for the training data.[1][2] As a result, unsupervised learning algorithms must first self-discover any naturally occurring patterns in that training data set. Common examples include clustering, where the algorithm automatically groups its training examples into categories with similar features, and principal component analysis, where the algorithm finds ways to compress the training data set by identifying which features are most useful for discriminating between different training examples, and discarding the rest. This contrasts with supervised learning in which the training data include pre-assigned category labels (often by a human, or from the output of non-learning classification algorithm). Other intermediate levels in the supervision spectrum include reinforcement learning, where only numerical scores are available for each training example instead of detailed tags, and semi-supervised learning where only a portion of the training data have been tagged. | Semi-supervised learning is an approach to machine learning that combines a small amount of labeled data with a large amount of unlabeled data during training. Semi-supervised learning falls between unsupervised learning (with no labeled training data) and supervised learning (with only labeled training data). It is a special instance of weak supervision. | Reinforcement learning (RL) is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning. | Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.",
      "topics": [
        {
          "topic_id": 0,
          "count": 5,
          "phrases": [
            {
              "text": "technology",
              "score": 0.1007437481087233
            },
            {
              "text": "microsoft",
              "score": 0.08701055137154726
            },
            {
              "text": "company",
              "score": 0.07973424604917556
            },
            {
              "text": "apple",
              "score": 0.07213985044118004
            },
            {
              "text": "companies",
              "score": 0.06418162980264484
            },
            {
              "text": "amazon",
              "score": 0.05579841684222883
            },
            {
              "text": "multinational",
              "score": 0.05579841684222883
            },
            {
              "text": "software",
              "score": 0.05579841684222883
            },
            {
              "text": "revenue",
              "score": 0.04690415023839433
            },
            {
              "text": "inc",
              "score": 0.04690415023839433
            }
          ]
        },
        {
          "topic_id": 1,
          "count": 5,
          "phrases": [
            {
              "text": "learning",
              "score": 0.18309249415991027
            },
            {
              "text": "training",
              "score": 0.11836015401314076
            },
            {
              "text": "data",
              "score": 0.10109415817647079
            },
            {
              "text": "supervised",
              "score": 0.09495941807377267
            },
            {
              "text": "algorithm",
              "score": 0.0751562032138162
            },
            {
              "text": "machine",
              "score": 0.06049656014890447
            },
            {
              "text": "unsupervised",
              "score": 0.05259467999004345
            },
            {
              "text": "labeled",
              "score": 0.04421108898068748
            },
            {
              "text": "labels",
              "score": 0.03522302060478677
            },
            {
              "text": "input",
              "score": 0.03522302060478677
            }
          ]
        }
      ],
      "topic_assignments": [
        {
          "text": "Google LLC is an American multinational technology company that specializes in Internet-related services and products, which include online advertising technologies, a search engine, cloud computing, software, and hardware. It is considered one of the Big Five companies in the American information technology industry, along with Amazon, Apple, Meta (Facebook) and Microsoft.",
          "assigned_id": 0,
          "probabilities": [
            1,
            4.085054396619016e-309
          ]
        },
        {
          "text": "Amazon.com, Inc. is an American multinational technology company which focuses on e-commerce, cloud computing, digital streaming, and artificial intelligence. It is one of the Big Five companies in the U.S. information technology industry, along with Google (Alphabet), Apple, Meta (Facebook), and Microsoft. The company has been referred to as one of the most influential economic and cultural forces in the world, as well as the world's most valuable brand.",
          "assigned_id": 0,
          "probabilities": [
            1,
            4.60356554840347e-309
          ]
        },
        {
          "text": "Meta Platforms, Inc., doing business as Meta and formerly known as Facebook, Inc., is a multinational technology conglomerate based in Menlo Park, California. The company is the parent organization of Facebook, Instagram, and WhatsApp, among other subsidiaries. Meta is one of the world's most valuable companies and is considered one of the Big Tech companies in U.S. information technology, alongside Amazon, Google, Apple, and Microsoft. The company generates a substantial share of its revenue from the sale of advertisement placements to marketers.",
          "assigned_id": 0,
          "probabilities": [
            1,
            4.070943934486963e-309
          ]
        },
        {
          "text": "Apple Inc. is an American multinational technology company that specializes in consumer electronics, computer software and online services. Apple is the largest information technology company by revenue (totaling $274.5 billion in 2020) and, since January 2021, the world's most valuable company. As of 2021, Apple is the fourth-largest PC vendor by unit sales and fourth-largest smartphone manufacturer. It is one of the Big Five American information technology companies, alongside Amazon, Google (Alphabet), Facebook (Meta), and Microsoft.",
          "assigned_id": 0,
          "probabilities": [
            0.6053796529377782,
            0.1948665243301664
          ]
        },
        {
          "text": "Microsoft Corporation is an American multinational technology corporation which produces computer software, consumer electronics, personal computers, and related services. Its best-known software products are the Microsoft Windows line of operating systems, the Microsoft Office suite, and the Internet Explorer and Edge web browsers. Its flagship hardware products are the Xbox video game consoles and the Microsoft Surface lineup of touchscreen personal computers. Microsoft ranked No. 21 in the 2020 Fortune 500 rankings of the largest United States corporations by total revenue; it was the world's largest software maker by revenue as of 2016. It is considered one of the Big Five companies in the U.S. information technology industry, along with Amazon, Google (Alphabet), Apple, and Facebook (Meta).",
          "assigned_id": 0,
          "probabilities": [
            1,
            3.565365632376766e-309
          ]
        },
        {
          "text": "Supervised learning (SL) is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples. In supervised learning, each example is a pair consisting of an input object (typically a vector) and a desired output value (also called the supervisory signal). A supervised learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples. An optimal scenario will allow for the algorithm to correctly determine the class labels for unseen instances. This requires the learning algorithm to generalize from the training data to unseen situations in a reasonable way (see inductive bias). This statistical quality of an algorithm is measured through the so-called generalization error.",
          "assigned_id": 1,
          "probabilities": [
            0.15511752359976377,
            0.6804084692327057
          ]
        },
        {
          "text": "Unsupervised learning is a type of machine learning in which the algorithm is not provided with any pre-assigned labels or scores for the training data.[1][2] As a result, unsupervised learning algorithms must first self-discover any naturally occurring patterns in that training data set. Common examples include clustering, where the algorithm automatically groups its training examples into categories with similar features, and principal component analysis, where the algorithm finds ways to compress the training data set by identifying which features are most useful for discriminating between different training examples, and discarding the rest. This contrasts with supervised learning in which the training data include pre-assigned category labels (often by a human, or from the output of non-learning classification algorithm). Other intermediate levels in the supervision spectrum include reinforcement learning, where only numerical scores are available for each training example instead of detailed tags, and semi-supervised learning where only a portion of the training data have been tagged.",
          "assigned_id": 1,
          "probabilities": [
            4.048049922344117e-309,
            1
          ]
        },
        {
          "text": "Semi-supervised learning is an approach to machine learning that combines a small amount of labeled data with a large amount of unlabeled data during training. Semi-supervised learning falls between unsupervised learning (with no labeled training data) and supervised learning (with only labeled training data). It is a special instance of weak supervision.",
          "assigned_id": 1,
          "probabilities": [
            3.83119492212747e-309,
            1
          ]
        },
        {
          "text": "Reinforcement learning (RL) is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning.",
          "assigned_id": 1,
          "probabilities": [
            0.17342063980801026,
            0.6442718319782724
          ]
        },
        {
          "text": "Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.",
          "assigned_id": 1,
          "probabilities": [
            4.556125110870383e-309,
            1
          ]
        }
      ]
    }
  ]
}
```

### Component configuration
This is a component wrapped on the basis of [BERTopic](https://github.com/MaartenGr/BERTopic).

| Parameter | Type | Default value | Description |
| --- | --- | --- | --- |
| top_n_words | int | 10 | The number of words per topic to extract. Setting this too high can negatively impact topic embeddings as topics are typically best represented by at most 10 words. |
| n_gram_range | Tuple[int, int] | [1,1] | The n-gram range for the CountVectorizer. Advised to keep high values between 1 and 3. More would likely lead to memory issues. |
| min_topic_size | int | 3 | The minimum size of the topic. Increasing this value will lead to a lower number of clusters/topics. It tells HDBSCAN what the minimum size of a cluster should be before it is accepted as a cluster. When you set this parameter very high, you will get very few clusters as they all need to be high. In contrast, if you set this too low you might end with too many extremely specific clusters. |
| nr_topics | Union[int, str] | null | Each resulting topic has its own feature vector constructed from c-TF-IDF. Using those feature vectors, we can find the most similar topics and merge them. If we do this iteratively, starting from the least frequent topic, we can reduce the number of topics quite easily. We do this until we reach the value of nr_topics. Use "auto" to automatically reduce topics that have a similarity of at least 0.9, do not maps all others.|
| low_memory | bool | false | Sets UMAP low memory to True to make sure less memory is used. |
| seed_topic_list | List[List[str]] | null | A list of seed words per topic to converge around. see the [Guided Topic Modeling](https://maartengr.github.io/BERTopic/tutorial/guided/guided.html). |
| verbose | bool | false | Changes the verbosity of the model, Set to True if you want to track the stages of the model. |

Component's config can be modified in [`components/config.cfg`](components/config.cfg) for default values, or on the per API request basis at runtime:

```json
{
  "articles": [
    {
      "text": "Google LLC is an American multinational technology company that specializes in Internet-related services and products, which include online advertising technologies, a search engine, cloud computing, software, and hardware. It is considered one of the Big Five companies in the American information technology industry, along with Amazon, Apple, Meta (Facebook) and Microsoft. | Amazon.com, Inc. is an American multinational technology company which focuses on e-commerce, cloud computing, digital streaming, and artificial intelligence. It is one of the Big Five companies in the U.S. information technology industry, along with Google (Alphabet), Apple, Meta (Facebook), and Microsoft. The company has been referred to as one of the most influential economic and cultural forces in the world, as well as the world's most valuable brand. | Meta Platforms, Inc., doing business as Meta and formerly known as Facebook, Inc., is a multinational technology conglomerate based in Menlo Park, California. The company is the parent organization of Facebook, Instagram, and WhatsApp, among other subsidiaries. Meta is one of the world's most valuable companies and is considered one of the Big Tech companies in U.S. information technology, alongside Amazon, Google, Apple, and Microsoft. The company generates a substantial share of its revenue from the sale of advertisement placements to marketers. | Apple Inc. is an American multinational technology company that specializes in consumer electronics, computer software and online services. Apple is the largest information technology company by revenue (totaling $274.5 billion in 2020) and, since January 2021, the world's most valuable company. As of 2021, Apple is the fourth-largest PC vendor by unit sales and fourth-largest smartphone manufacturer. It is one of the Big Five American information technology companies, alongside Amazon, Google (Alphabet), Facebook (Meta), and Microsoft. | Microsoft Corporation is an American multinational technology corporation which produces computer software, consumer electronics, personal computers, and related services. Its best-known software products are the Microsoft Windows line of operating systems, the Microsoft Office suite, and the Internet Explorer and Edge web browsers. Its flagship hardware products are the Xbox video game consoles and the Microsoft Surface lineup of touchscreen personal computers. Microsoft ranked No. 21 in the 2020 Fortune 500 rankings of the largest United States corporations by total revenue; it was the world's largest software maker by revenue as of 2016. It is considered one of the Big Five companies in the U.S. information technology industry, along with Amazon, Google (Alphabet), Apple, and Facebook (Meta). | Supervised learning (SL) is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples. In supervised learning, each example is a pair consisting of an input object (typically a vector) and a desired output value (also called the supervisory signal). A supervised learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples. An optimal scenario will allow for the algorithm to correctly determine the class labels for unseen instances. This requires the learning algorithm to generalize from the training data to unseen situations in a reasonable way (see inductive bias). This statistical quality of an algorithm is measured through the so-called generalization error. | Unsupervised learning is a type of machine learning in which the algorithm is not provided with any pre-assigned labels or scores for the training data.[1][2] As a result, unsupervised learning algorithms must first self-discover any naturally occurring patterns in that training data set. Common examples include clustering, where the algorithm automatically groups its training examples into categories with similar features, and principal component analysis, where the algorithm finds ways to compress the training data set by identifying which features are most useful for discriminating between different training examples, and discarding the rest. This contrasts with supervised learning in which the training data include pre-assigned category labels (often by a human, or from the output of non-learning classification algorithm). Other intermediate levels in the supervision spectrum include reinforcement learning, where only numerical scores are available for each training example instead of detailed tags, and semi-supervised learning where only a portion of the training data have been tagged. | Semi-supervised learning is an approach to machine learning that combines a small amount of labeled data with a large amount of unlabeled data during training. Semi-supervised learning falls between unsupervised learning (with no labeled training data) and supervised learning (with only labeled training data). It is a special instance of weak supervision. | Reinforcement learning (RL) is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning. | Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised."
    }
  ],
  "component_cfg": {
    "topic": {"top_n_words": 5, "n_gram_range": [1,2]}
  }
}
```

```json
{
  "topic": [
    {
      "text": "Google LLC is an American multinational technology company that specializes in Internet-related services and products, which include online advertising technologies, a search engine, cloud computing, software, and hardware. It is considered one of the Big Five companies in the American information technology industry, along with Amazon, Apple, Meta (Facebook) and Microsoft. | Amazon.com, Inc. is an American multinational technology company which focuses on e-commerce, cloud computing, digital streaming, and artificial intelligence. It is one of the Big Five companies in the U.S. information technology industry, along with Google (Alphabet), Apple, Meta (Facebook), and Microsoft. The company has been referred to as one of the most influential economic and cultural forces in the world, as well as the world's most valuable brand. | Meta Platforms, Inc., doing business as Meta and formerly known as Facebook, Inc., is a multinational technology conglomerate based in Menlo Park, California. The company is the parent organization of Facebook, Instagram, and WhatsApp, among other subsidiaries. Meta is one of the world's most valuable companies and is considered one of the Big Tech companies in U.S. information technology, alongside Amazon, Google, Apple, and Microsoft. The company generates a substantial share of its revenue from the sale of advertisement placements to marketers. | Apple Inc. is an American multinational technology company that specializes in consumer electronics, computer software and online services. Apple is the largest information technology company by revenue (totaling $274.5 billion in 2020) and, since January 2021, the world's most valuable company. As of 2021, Apple is the fourth-largest PC vendor by unit sales and fourth-largest smartphone manufacturer. It is one of the Big Five American information technology companies, alongside Amazon, Google (Alphabet), Facebook (Meta), and Microsoft. | Microsoft Corporation is an American multinational technology corporation which produces computer software, consumer electronics, personal computers, and related services. Its best-known software products are the Microsoft Windows line of operating systems, the Microsoft Office suite, and the Internet Explorer and Edge web browsers. Its flagship hardware products are the Xbox video game consoles and the Microsoft Surface lineup of touchscreen personal computers. Microsoft ranked No. 21 in the 2020 Fortune 500 rankings of the largest United States corporations by total revenue; it was the world's largest software maker by revenue as of 2016. It is considered one of the Big Five companies in the U.S. information technology industry, along with Amazon, Google (Alphabet), Apple, and Facebook (Meta). | Supervised learning (SL) is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples. In supervised learning, each example is a pair consisting of an input object (typically a vector) and a desired output value (also called the supervisory signal). A supervised learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples. An optimal scenario will allow for the algorithm to correctly determine the class labels for unseen instances. This requires the learning algorithm to generalize from the training data to unseen situations in a reasonable way (see inductive bias). This statistical quality of an algorithm is measured through the so-called generalization error. | Unsupervised learning is a type of machine learning in which the algorithm is not provided with any pre-assigned labels or scores for the training data.[1][2] As a result, unsupervised learning algorithms must first self-discover any naturally occurring patterns in that training data set. Common examples include clustering, where the algorithm automatically groups its training examples into categories with similar features, and principal component analysis, where the algorithm finds ways to compress the training data set by identifying which features are most useful for discriminating between different training examples, and discarding the rest. This contrasts with supervised learning in which the training data include pre-assigned category labels (often by a human, or from the output of non-learning classification algorithm). Other intermediate levels in the supervision spectrum include reinforcement learning, where only numerical scores are available for each training example instead of detailed tags, and semi-supervised learning where only a portion of the training data have been tagged. | Semi-supervised learning is an approach to machine learning that combines a small amount of labeled data with a large amount of unlabeled data during training. Semi-supervised learning falls between unsupervised learning (with no labeled training data) and supervised learning (with only labeled training data). It is a special instance of weak supervision. | Reinforcement learning (RL) is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning. | Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.",
      "topics": [
        {
          "topic_id": 0,
          "count": 5,
          "phrases": [
            {
              "text": "microsoft",
              "score": 0.051473178260499027
            },
            {
              "text": "apple",
              "score": 0.04227001948592211
            },
            {
              "text": "information technology",
              "score": 0.03740667281997986
            },
            {
              "text": "companies",
              "score": 0.03740667281997986
            },
            {
              "text": "multinational technology",
              "score": 0.03233055998526882
            }
          ]
        },
        {
          "topic_id": 1,
          "count": 5,
          "phrases": [
            {
              "text": "supervised",
              "score": 0.056651617478428104
            },
            {
              "text": "training data",
              "score": 0.05264290971402835
            },
            {
              "text": "supervised learning",
              "score": 0.048514241067081125
            },
            {
              "text": "semi supervised",
              "score": 0.02544658056524801
            },
            {
              "text": "unsupervised learning",
              "score": 0.02544658056524801
            }
          ]
        }
      ],
      "topic_assignments": [
        {
          "text": "Google LLC is an American multinational technology company that specializes in Internet-related services and products, which include online advertising technologies, a search engine, cloud computing, software, and hardware. It is considered one of the Big Five companies in the American information technology industry, along with Amazon, Apple, Meta (Facebook) and Microsoft.",
          "assigned_id": 0,
          "probabilities": [
            1,
            4.00915414113993e-309
          ]
        },
        {
          "text": "Amazon.com, Inc. is an American multinational technology company which focuses on e-commerce, cloud computing, digital streaming, and artificial intelligence. It is one of the Big Five companies in the U.S. information technology industry, along with Google (Alphabet), Apple, Meta (Facebook), and Microsoft. The company has been referred to as one of the most influential economic and cultural forces in the world, as well as the world's most valuable brand.",
          "assigned_id": 0,
          "probabilities": [
            0.733794511643719,
            0.21171734249991556
          ]
        },
        {
          "text": "Meta Platforms, Inc., doing business as Meta and formerly known as Facebook, Inc., is a multinational technology conglomerate based in Menlo Park, California. The company is the parent organization of Facebook, Instagram, and WhatsApp, among other subsidiaries. Meta is one of the world's most valuable companies and is considered one of the Big Tech companies in U.S. information technology, alongside Amazon, Google, Apple, and Microsoft. The company generates a substantial share of its revenue from the sale of advertisement placements to marketers.",
          "assigned_id": 0,
          "probabilities": [
            1,
            3.82561181500552e-309
          ]
        },
        {
          "text": "Apple Inc. is an American multinational technology company that specializes in consumer electronics, computer software and online services. Apple is the largest information technology company by revenue (totaling $274.5 billion in 2020) and, since January 2021, the world's most valuable company. As of 2021, Apple is the fourth-largest PC vendor by unit sales and fourth-largest smartphone manufacturer. It is one of the Big Five American information technology companies, alongside Amazon, Google (Alphabet), Facebook (Meta), and Microsoft.",
          "assigned_id": 0,
          "probabilities": [
            0.7572607572766432,
            0.1860594394886937
          ]
        },
        {
          "text": "Microsoft Corporation is an American multinational technology corporation which produces computer software, consumer electronics, personal computers, and related services. Its best-known software products are the Microsoft Windows line of operating systems, the Microsoft Office suite, and the Internet Explorer and Edge web browsers. Its flagship hardware products are the Xbox video game consoles and the Microsoft Surface lineup of touchscreen personal computers. Microsoft ranked No. 21 in the 2020 Fortune 500 rankings of the largest United States corporations by total revenue; it was the world's largest software maker by revenue as of 2016. It is considered one of the Big Five companies in the U.S. information technology industry, along with Amazon, Google (Alphabet), Apple, and Facebook (Meta).",
          "assigned_id": 0,
          "probabilities": [
            1,
            3.63329776803299e-309
          ]
        },
        {
          "text": "Supervised learning (SL) is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples. In supervised learning, each example is a pair consisting of an input object (typically a vector) and a desired output value (also called the supervisory signal). A supervised learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples. An optimal scenario will allow for the algorithm to correctly determine the class labels for unseen instances. This requires the learning algorithm to generalize from the training data to unseen situations in a reasonable way (see inductive bias). This statistical quality of an algorithm is measured through the so-called generalization error.",
          "assigned_id": 1,
          "probabilities": [
            3.692993276574767e-309,
            1
          ]
        },
        {
          "text": "Unsupervised learning is a type of machine learning in which the algorithm is not provided with any pre-assigned labels or scores for the training data.[1][2] As a result, unsupervised learning algorithms must first self-discover any naturally occurring patterns in that training data set. Common examples include clustering, where the algorithm automatically groups its training examples into categories with similar features, and principal component analysis, where the algorithm finds ways to compress the training data set by identifying which features are most useful for discriminating between different training examples, and discarding the rest. This contrasts with supervised learning in which the training data include pre-assigned category labels (often by a human, or from the output of non-learning classification algorithm). Other intermediate levels in the supervision spectrum include reinforcement learning, where only numerical scores are available for each training example instead of detailed tags, and semi-supervised learning where only a portion of the training data have been tagged.",
          "assigned_id": 1,
          "probabilities": [
            0.18799752163567685,
            0.734198392900434
          ]
        },
        {
          "text": "Semi-supervised learning is an approach to machine learning that combines a small amount of labeled data with a large amount of unlabeled data during training. Semi-supervised learning falls between unsupervised learning (with no labeled training data) and supervised learning (with only labeled training data). It is a special instance of weak supervision.",
          "assigned_id": 1,
          "probabilities": [
            3.752850889125207e-309,
            1
          ]
        },
        {
          "text": "Reinforcement learning (RL) is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning.",
          "assigned_id": 1,
          "probabilities": [
            0.1826140167701155,
            0.7902966552339149
          ]
        },
        {
          "text": "Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.",
          "assigned_id": 1,
          "probabilities": [
            4.01932930926635e-309,
            1
          ]
        }
      ]
    }
  ]
}
```

### Implementation details
This repository only intergrates the [BERTopic](https://github.com/dmmiller612/bert-extractive-summarizer) with [Sentence Transformers](https://www.sbert.net/) backend, the usage of topic visualization, search topics, topic per class, (semi)-supervised topic modeling, custom sub-models, and all the operations after training are not implemented.