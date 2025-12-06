import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from clean import cleanFull

def price_heatmap():
    df, _ = cleanFull() 
    plt.figure(figsize=(10,7))
    sns.kdeplot(
        data=df,
        x="longitude",
        y="latitude",
        cmap="hot",
        fill=True,
        levels=200,
        thresh=0.05
    )
    plt.title("NYC Airbnb Price Density Heatmap")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

if __name__ == "__main__":
    price_heatmap()
