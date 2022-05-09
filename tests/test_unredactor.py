import unredactor


def test_fetch_data():
    df = unredactor.fetch_data()
    assert not df.empty


def test_setup_training_data():
    df = unredactor.fetch_data()
    cdf = unredactor.clean_data(df)
    train_df, test_df = unredactor.setup_training_data(cdf)
    assert not (train_df.empty and test_df.empty)


def test_train_and_predict():
    df = unredactor.fetch_data()
    cdf = unredactor.clean_data(df)
    train_df, test_df = unredactor.setup_training_data(cdf)
    p_score, r_score, f_score = unredactor.train_and_predict(train_df, test_df)
    p_flag, r_flag, f_flag = p_score < 1, r_score < 1, f_score < 1
    assert p_flag and r_flag and f_flag
