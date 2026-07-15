# AlphaDGO



## 📦Data Preparation
To prepare the raw data, refer to [AlphaForge](https://github.com/dulyhao/alphaforge) , which uses [Qlib](https://github.com/microsoft/qlib) as the data backend.

Before running the data fetching script, make sure to configure the following paths:

- Set `qlib_base_data_path` in `data_collection/fetch_baostock_data.py`.
- Set `QLIB_PATH` in `gan/utils/data.py`

Then execute:

```shell
python fetch_baostock_data.py
```



## ⛏️Alpha Mining
The alpha mining process begins with training the FLC (Financial-Logic Critic) model:

```shell
python llm_labeling/train_expression_regressor.py
```

After training, configure your LLM connection settings by modifying the following variables in  `alpha_mining.py`:

- `model_name`
- `model_base_url`
- `model_api_key`

Then run the alpha mining pipeline:

```shell
python alpha_mining.py
```



## 🔗Alpha Combine

To combine the generated alpha factors, use:

```shell
python alpha_combine.py
```



## 🧪Test

Run the evaluation script to test the performance of the generated alphas:

```shell
python model_test.py
```

