import geopandas as gpd
import pandas as pd
import networkx as nx
import random
from shapely.geometry import LineString
import sys
import config


def main():
    # =====================================================
    # 1. INPUT DATE
    # =====================================================

    input_date = input("Enter evacuation date (DD-MM-YYYY): ").strip()
    print("\nSelected Date:", input_date)

    # =====================================================
    # 2. LOAD DATA
    # =====================================================

    # prefer existing shapefile if gpkg not present
    roads_path = config.gis("Chennai_Roads.gpkg")
    if not roads_path.exists():
        roads_path = config.gis("Chennai_Roads.shp")

    zones_path = config.gis("flood_risk_zones.gpkg")
    if not zones_path.exists():
        zones_path = config.gis("flood_risk_zones.shp")

    roads = gpd.read_file(str(roads_path))
    zones_geom = gpd.read_file(str(zones_path))
    daily_data = pd.read_csv(str(config.processed("final_zone_ml_dataset.csv")))

    print("Files loaded successfully!")

    # =====================================================
    # 3. PREPARE DATE DATA
    # =====================================================

    daily_data["date"] = pd.to_datetime(daily_data["date"], dayfirst=True)
    selected_date = pd.to_datetime(input_date, dayfirst=True)

    daily_filtered = daily_data[daily_data["date"] == selected_date]

    if daily_filtered.empty:
        print("No data available for this date.")
        sys.exit()

    print("Zones for this date:", len(daily_filtered))

    # =====================================================
    # 4. MERGE ML DATA WITH ZONES
    # =====================================================

    zones = zones_geom.merge(
        daily_filtered[["zone_id", "predicted_runoff_mm", "flood_label"]],
        on="zone_id",
        how="left"
    )

    zones["predicted_runoff_mm"] = zones["predicted_runoff_mm"].fillna(0)
    zones["flood_label"] = zones["flood_label"].fillna(0)

    # =====================================================
    # 5. SPATIAL JOIN WITH ROADS
    # =====================================================

    roads = gpd.sjoin(
        roads,
        zones[["zone_id", "predicted_runoff_mm", "flood_label", "geometry"]],
        how="left",
        predicate="intersects"
    )

    roads["predicted_runoff_mm"] = roads["predicted_runoff_mm"].fillna(0)
    roads["flood_label"] = roads["flood_label"].fillna(0)

    # =====================================================
    # 6. CALCULATE EVACUATION COST
    # =====================================================

    roads["evac_cost"] = roads.length * (1 + roads["predicted_runoff_mm"])

    # heavily penalize flooded roads
    roads.loc[roads["flood_label"] == 1, "evac_cost"] = 999999

    print("Evacuation cost updated based on flood prediction.")

    # =====================================================
    # 7. BUILD GRAPH
    # =====================================================

    G = nx.Graph()

    for idx, row in roads.iterrows():

        geom = row.geometry
        cost = row["evac_cost"]

        if geom is None:
            continue

        if geom.geom_type == "MultiLineString":
            lines = geom.geoms
        else:
            lines = [geom]

        for line in lines:

            start = tuple(line.coords[0])
            end = tuple(line.coords[-1])

            G.add_edge(start, end, weight=cost)

    print("Graph created.")
    print("Total nodes:", len(G.nodes))

    # =====================================================
    # 8. FIND LARGEST CONNECTED COMPONENT
    # =====================================================

    largest_cc = max(nx.connected_components(G), key=len)
    cc_nodes = list(largest_cc)

    print("Largest connected component size:", len(cc_nodes))

    # =====================================================
    # 9. SELECT FLOOD AND SAFE NODES
    # =====================================================

    flood_nodes = []
    safe_nodes = []

    for idx, row in roads.iterrows():

        geom = row.geometry
        flood = row["flood_label"]

        if geom is None:
            continue

        if geom.geom_type == "MultiLineString":
            lines = geom.geoms
        else:
            lines = [geom]

        for line in lines:

            start = tuple(line.coords[0])
            end = tuple(line.coords[-1])

            if start in cc_nodes and end in cc_nodes:

                if flood == 1:
                    flood_nodes.extend([start, end])
                else:
                    safe_nodes.extend([start, end])

    flood_nodes = list(set(flood_nodes))
    safe_nodes = list(set(safe_nodes))

    print("Flood candidate nodes:", len(flood_nodes))
    print("Safe candidate nodes:", len(safe_nodes))

    if not flood_nodes or not safe_nodes:
        print("Not enough nodes for routing.")
        sys.exit()

    start_node = random.choice(flood_nodes)
    end_node = random.choice(safe_nodes)

    print("\nStart node (flooded area):", start_node)
    print("End node (safe area):", end_node)

    # =====================================================
    # 10. RUN DIJKSTRA
    # =====================================================

    try:

        path = nx.shortest_path(G, source=start_node, target=end_node, weight="weight")
        total_cost = nx.shortest_path_length(G, source=start_node, target=end_node, weight="weight")

        print("\nEvacuation path found!")
        print("Total evacuation cost:", total_cost)
        print("Nodes in path:", len(path))

    except nx.NetworkXNoPath:

        print("No path available between selected nodes.")
        sys.exit()

    # =====================================================
    # 11. EXPORT ROUTE
    # =====================================================

    route_lines = []

    for i in range(len(path) - 1):
        route_lines.append(LineString([path[i], path[i + 1]]))

    route_gdf = gpd.GeoDataFrame(geometry=route_lines, crs=roads.crs)

    output_file = str(config.results(f"evacuation_route_{input_date.replace('-','')}.gpkg"))

    route_gdf.to_file(output_file, driver="GPKG")

    print("\nRoute exported successfully!")
    print("File:", output_file)


if __name__ == "__main__":
    main()