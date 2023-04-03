from flow.trial_runner import TrialRunner

if __name__ == '__main__':
    s = TrialRunner("trial_runner_conf.json")
    s.fit()
    s.to_csv("score_df.csv")
    s.plt_result("score_df.csv")

    print(s.get_dataframe())
