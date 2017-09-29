# h2o-model-explanation

Installing the environment:

```sh
# using this config
conda env create -f environment.yml 
# from scratch
conda create --name <name_of_env>
```

Update environment configuration

```sh
conda env export > environment.yml
```

Force update of an environment

```sh
conda env update -f environment.yml
```

