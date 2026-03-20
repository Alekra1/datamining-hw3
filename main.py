import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics, tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("scotch.csv")

out_df = df.iloc[-3:].reset_index(drop=True)

df = df.iloc[:-3]


df["Region"] = df.apply(lambda row: row["islay":"islands"].idxmax(), axis=1)

df = df.drop(columns=df.loc[:, "islay":"islands"])

X = df.loc[:, "wyne":"quick"]
y = df["Region"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


if __name__ == "__main__":
    # print(X)
    # print(y)
    dtree = DecisionTreeClassifier(max_depth=4, min_samples_split=50)
    dtree = dtree.fit(X_train, y_train)

    y_pred = dtree.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    X_pred = out_df.loc[:, "wyne":"quick"]

    y_result = dtree.predict(X_pred)
    pred_prob = dtree.predict_proba(X_pred)

    print("\nPredictions:")
    for i, pred in enumerate(y_result):
        print(f"NEW {i + 1} predicted as: {pred}")

    decision_tree_results = pd.DataFrame(columns=dtree.classes_, data=pred_prob)
    decision_tree_results.insert(loc=0, column="NAME", value=["NEW1", "NEW2", "NEW3"])
    print(decision_tree_results)

    plt.figure(figsize=(20, 10))
    tree.plot_tree(
        dtree, feature_names=X.columns, class_names=dtree.classes_, filled=True
    )

    # text_representation = tree.export_text(dtree, feature_names=list(X.columns))
    # print(text_representation)

    plt.savefig("dtree.png")
