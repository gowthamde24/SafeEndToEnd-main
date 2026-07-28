import carla
import csv

def generate_town04_waypoints():
    # 1. Connect to CARLA and load Town04
    client = carla.Client('192.168.56.1', 2000)
    client.set_timeout(10.0)
    world = client.load_world('Town04')
    carla_map = world.get_map()

    # 2. Get the exact spawn point you use in run_iter.py (PLAYER_START_INDEX = 2)
    spawn_points = carla_map.get_spawn_points()
    start_transform = spawn_points[2]

    # 3. Snap to the nearest waypoint in the center of the lane
    current_waypoint = carla_map.get_waypoint(start_transform.location)

    waypoints_list = []
    
    # 4. Walk forward along the lane for 5000 meters, saving the coordinates every 1 meter
    print("Generating waypoints...")
    for _ in range(5000):
        waypoints_list.append([current_waypoint.transform.location.x, current_waypoint.transform.location.y])
        
        # Get the next waypoint 1.0 meter ahead
        next_waypoints = current_waypoint.next(1.0)
        if not next_waypoints:
            break # Reached the end of the road
        current_waypoint = next_waypoints[0]

    # 5. Save the coordinates to a text file
    filename = 'town04_waypoints.txt'
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(waypoints_list)
        
    print(f"Success! Generated {len(waypoints_list)} waypoints and saved to {filename}")

if __name__ == '__main__':
    generate_town04_waypoints()