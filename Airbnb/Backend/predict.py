def predict_price(pre, rf, lr, nn, df):
    X = pre.transform(df)
    dense = X.toarray() if hasattr(X,"toarray") else X
    rf_p = float(rf.predict(X)[0])
    lr_p = float(lr.predict(X)[0])
    nn_p = float(nn.predict(dense)[0][0])
    return {
        "random_forest": rf_p,
        "linear_regression": lr_p,
        "neural_network": nn_p,
        "final_price": rf_p,
        "confidence_low": rf_p*0.9,
        "confidence_high": rf_p*1.1
    }
