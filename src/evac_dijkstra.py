import geopandas as gpd
import networkx as nx
import random
from shapely.geometry import LineString, Point

# ============================================
# 1️⃣ LOAD DATA
# ============================================

roads = gpd.read_file("finalevac.gpkg")
zones = gpd.read_file("flood_risk_zones.gpkg")

print("Road file loaded successfully!")
print("Zone file loaded successfully!")

# Ensure same CRS
if roads.crs != zones.crs:
    zones = zones.to_crs(roads.crs)

# ============================================
# 2️⃣ BUILD GRAPH
# ============================================

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

print("Graph created successfully!")
print("Total nodes:", len(G.nodes))
print("Total edges:", len(G.edges))

# ============================================
# 3️⃣ LARGEST CONNECTED COMPONENT
# ============================================

largest_cc = max(nx.connected_components(G), key=len)
largest_cc = set(largest_cc)

print("Largest connected component size:", len(largest_cc))

# ============================================
# 4️⃣ SELECT HIGH → LOW NODES (CONNECTED ONLY)
# ============================================

high_risk_nodes = []
low_risk_nodes = []

for idx, row in roads.iterrows():
    geom = row.geometry
    risk = row["risk_class"]

    if geom is None:
        continue

    if geom.geom_type == "MultiLineString":
        lines = geom.geoms
    else:
        lines = [geom]

    for line in lines:
        start = tuple(line.coords[0])
        end = tuple(line.coords[-1])

        if start in largest_cc and end in largest_cc:

            if risk == "High":
                high_risk_nodes.append(start)
                high_risk_nodes.append(end)

            elif risk == "Low":
                low_risk_nodes.append(start)
                low_risk_nodes.append(end)

print("High risk candidate nodes (connected):", len(high_risk_nodes))
print("Low risk candidate nodes (connected):", len(low_risk_nodes))

if len(high_risk_nodes) == 0 or len(low_risk_nodes) == 0:
    print("Not enough valid nodes for evacuation.")
    exit()

start_node = random.choice(high_risk_nodes)
end_node = random.choice(low_risk_nodes)

print("\nSelected Nodes:")
print("Start (HIGH risk):", start_node)
print("End (LOW risk):", end_node)

# ============================================
# 5️⃣ FIND START & END ZONES
# ============================================

start_point = Point(start_node)
end_point = Point(end_node)

start_zone = zones[zones.contains(start_point)]
end_zone = zones[zones.contains(end_point)]

print("\nZone Information:")

if not start_zone.empty:
    print("Start Zone ID:", start_zone.iloc[0]["zone_id"])
    print("Start Zone Risk:", start_zone.iloc[0]["risk_class"])
else:
    print("Start zone not found")

if not end_zone.empty:
    print("End Zone ID:", end_zone.iloc[0]["zone_id"])
    print("End Zone Risk:", end_zone.iloc[0]["risk_class"])
else:
    print("End zone not found")

# ============================================
# 6️⃣ RUN DIJKSTRA
# ============================================

try:
    path = nx.shortest_path(G, source=start_node, target=end_node, weight="weight")
    total_cost = nx.shortest_path_length(G, source=start_node, target=end_node, weight="weight")

    print("\nEvacuation path found!")
    print("Total evacuation cost:", total_cost)
    print("Number of nodes in path:", len(path))

except nx.NetworkXNoPath:
    print("No evacuation path found.")
    exit()

# ============================================
# 7️⃣ EXPORT ROUTE
# ============================================

path_lines = []

for i in range(len(path) - 1):
    line = LineString([path[i], path[i+1]])
    path_lines.append(line)

route_gdf = gpd.GeoDataFrame(geometry=path_lines, crs=roads.crs)

route_gdf.to_file("final_risk_based_evacuation_route.gpkg", driver="GPKG")

print("\nFinal evacuation route exported successfully!")
print("File: final_risk_based_evacuation_route.gpkg")