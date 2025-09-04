import argparse


def time_of_ball():
    parser = argparse.ArgumentParser(description="Calculate the time it takes for a ball to hit the ground from a given height on different planets.")
    parser.add_argument('height', type=float, help="height of object above ground")
    parser.add_argument('-gravity', choices=["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn","Uranus","Neptune"], help="Input planet name to use its gravity (in m/s²)")
    args = parser.parse_args()

    height = args.height

    planets_gravity = {
        "Mercury": 3.7,  # 3.7 m/s²
        "Venus": 8.87,  # 8.87 m/s²
        "Earth": 9.81,  # 9.81 m/s² (standard gravity)
        "Mars": 3.73,  # 3.73 m/s²
        "Jupiter": 24.79,  # 24.79 m/s²
        "Saturn": 10.44,  # 10.44 m/s²
        "Uranus": 8.69,  # 8.69 m/s²
        "Neptune": 11.15  # 11.15 m/s²
    }

    if args.gravity:
        gravity = planets_gravity[args.gravity]
        time = round(((2*height)/gravity)**(1/2), 3)
        return print("The time it takes for a ball to hit the ground assuming the gravity of "+ args.gravity+' '+"is" + ' ' + str(time))
    else:
        gravity = 9.81
        time = round(((2 * height) / gravity) ** (1 / 2), 3)
        return print("The time it takes for a ball to hit the ground assuming the gravity of the Earth is" + ' ' + str(time))


if __name__ == "__main__":
    time_of_ball()
