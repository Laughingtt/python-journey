from flow.task import ModelTask

if __name__ == '__main__':
    s = ModelTask("run_config.json")
    s.fit()
