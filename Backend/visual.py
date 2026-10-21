import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
from clean import cleanFull

def density_heatmap():
    df, _ = cleanFull()
    plt.figure(figsize=(12, 10))

    hb = plt.hexbin(
        df["longitude"], df["latitude"],
        gridsize=200, cmap="hot",
        bins="log", mincnt=1
    )

    plt.colorbar(hb, label="Density (log scale)")
    plt.title("NYC Airbnb Listing Density Heatmap (Hexbin)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


def density_with_points():
    df, _ = cleanFull()
    plt.figure(figsize=(12, 10))

    hb = plt.hexbin(
        df["longitude"], df["latitude"],
        gridsize=200, cmap="hot",
        bins="log", mincnt=1
    )

    plt.scatter(
        df["longitude"], df["latitude"],
        s=2, alpha=0.25, c="black"
    )

    plt.colorbar(hb, label="Density (log scale)")
    plt.title("NYC Airbnb Density + Raw Points")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

def cluster_plot():
    df, km = cleanFull()

    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=df, x="longitude", y="latitude",
        hue="geo_cluster", palette="tab20",
        s=8, alpha=0.7
    )

    plt.title("NYC Airbnb Geo-Clusters (KMeans-Based)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1))
    plt.show()



def average_price_heatmap():
    df, _ = cleanFull()
    plt.figure(figsize=(12, 10))

    hb = plt.hexbin(
        df["longitude"], df["latitude"],
        C=df["price"], reduce_C_function=np.mean,
        gridsize=200, cmap="viridis",
        mincnt=1
    )

    plt.colorbar(hb, label="Average Price")
    plt.title("NYC Airbnb Average Price Heatmap")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


def average_price_heatmap_map():
    # Load cleaned data
    df, _ = cleanFull()

    # GeoDataFrame conversion
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)  # convert to web mercator

    x = gdf.geometry.x
    y = gdf.geometry.y
    prices = gdf["price"].values

    fig, ax = plt.subplots(figsize=(14, 12))

    heat = ax.hexbin(
        x, y,
        C=prices,
        reduce_C_function=np.mean,
        gridsize=150,
        cmap="viridis",
        bins=None,
        mincnt=5,
        alpha=0.85
    )
    ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.DarkMatter,
        alpha=0.90
    )

    cbar = plt.colorbar(heat, ax=ax)
    cbar.set_label("Average Price ($)")

    ax.set_title("NYC Airbnb Average Price Heatmap (Hexbin Mean)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.show()

def interactive_density_map():
    df, _ = cleanFull()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    x = gdf.geometry.x
    y = gdf.geometry.y

    fig, ax = plt.subplots(figsize=(14, 12))

    density = ax.hexbin(
        x, y,
        gridsize=150,
        cmap="hot",
        bins="log",
        mincnt=1,
        alpha=0.85
    )

    points = ax.scatter(
        x, y,
        s=2, alpha=0.4, c="white",
        label="Listings"
    )

    # THE FREE DARK MAP THAT WORKS
    ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.DarkMatter,
        alpha=0.90
    )

    show_density = True
    show_points = True
    show_basemap = True

    def toggle(event):
        nonlocal show_density, show_points, show_basemap

        if event.key == "p":
            show_points = not show_points
            points.set_visible(show_points)
            fig.canvas.draw_idle()

        if event.key == "d":
            show_density = not show_density
            density.set_visible(show_density)
            fig.canvas.draw_idle()

        if event.key == "b":
            show_basemap = not show_basemap
            ax.images = []
            if show_basemap:
                ctx.add_basemap(
                    ax,
                    source=ctx.providers.CartoDB.DarkMatter,
                    alpha=0.90
                )
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", toggle)

    plt.colorbar(density, label="Density (log scale)")
    ax.legend(loc="upper right")

    ax.set_title("Interactive NYC Airbnb Density Map (P=Points, D=Density, B=Basemap)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.show()

if __name__ == "__main__":
    # density_heatmap()
    # density_with_points()
    # cluster_plot()
    # average_price_heatmap()
    average_price_heatmap_map()
    interactive_density_map()
