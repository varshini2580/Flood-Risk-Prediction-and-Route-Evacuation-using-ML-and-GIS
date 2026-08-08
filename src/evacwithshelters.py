import geopandas as gpd
import pandas as pd
import networkx as nx
from shapely.geometry import LineString
import random
import config


def main():
    # -------------------------------
    # LOAD DATA
    # -------------------------------

    roads_path = config.gis("Chennai_Roads.gpkg")
    if not roads_path.exists():
        roads_path = config.gis("Chennai_Roads.shp")

    roads = gpd.read_file(str(roads_path))
    zones = gpd.read_file(str(config.gis("zonesrisk.gpkg")))
    shelters = gpd.read_file(str(config.gis("shelter_safe.gpkg")))

    daily_data = pd.read_csv(str(config.processed("final_zone_ml_dataset.csv")))

    print("Files loaded successfully!")

    # -------------------------------
    # SELECT DATE
    # -------------------------------

    date = input("Enter evacuation date (DD-MM-YYYY): ")

    daily_filtered = daily_data[daily_data["date"] == date]

    if daily_filtered.empty:
        print("No data available for this date.")
        return

    print("Zones for this date:", len(daily_filtered))

    # -------------------------------
    # JOIN FLOOD DATA WITH ZONES
    # -------------------------------

    zones = zones.merge(
        daily_filtered[["zone_id", "flood_label"]],
        on="zone_id",
        how="left"
    )

    # -------------------------------
    # BUILD GRAPH
    # -------------------------------

    G = nx.Graph()

    for idx, row in roads.iterrows():

        geom = row.geometry
        length = geom.length

        if geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                start = tuple(line.coords[0])
                end = tuple(line.coords[-1])
                G.add_edge(start, end, weight=length)

        elif geom.geom_type == "LineString":
            start = tuple(geom.coords[0])
            end = tuple(geom.coords[-1])
            G.add_edge(start, end, weight=length)

    print("Graph created.")
    print("Total nodes:", len(G.nodes))

    # Keep only largest connected road network
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

    print("Largest connected component size:", len(G.nodes))

    # -------------------------------
    # SELECT FLOODED ZONES
    # -------------------------------

    flooded_zones = zones[zones["flood_label"] == 1]

    start_zone = flooded_zones.sample(1).iloc[0]
    start_point = start_zone.geometry.centroid
    start_node = min(G.nodes, key=lambda n: ((n[0]-start_point.x)**2 + (n[1]-start_point.y)**2))

    print("Start node:", start_node)

    # -------------------------------
    # FIND NEAREST SHELTER
    # -------------------------------

    shelter_nodes = []

    for idx,row in shelters.iterrows():

        point = row.geometry

        node = min(G.nodes, key=lambda n: ((n[0]-point.x)**2 + (n[1]-point.y)**2))
        shelter_nodes.append(node)

    end_node = random.choice(shelter_nodes)

    print("Shelter node:", end_node)

    # -------------------------------
    # DIJKSTRA PATH
    # -------------------------------

    try:

        path = nx.shortest_path(G, start_node, end_node, weight="weight")
        cost = nx.shortest_path_length(G, start_node, end_node, weight="weight")

        print("Evacuation path found!")
        print("Total evacuation cost:", cost)
        print("Nodes in path:", len(path))

        lines = []

        for i in range(len(path)-1):
            lines.append(LineString([path[i], path[i+1]]))

        route = gpd.GeoDataFrame(geometry=lines, crs=roads.crs)

        output = str(config.results(f"EvacuationRoute_{date.replace('-','')}.gpkg"))

        route.to_file(output, driver="GPKG")

        print("Route exported successfully!")
        print("File:", output)

    except Exception:

        print("No path available.")


if __name__ == "__main__":
    main()