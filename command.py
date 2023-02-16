# import the necessary packages
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=False,
	help="name of the user")
args = vars(ap.parse_args())

# prints dictionary
# print(args)

# display a friendly message to the user
if args["name"] != None:
    print("Hi there {}, it's nice to meet you!".format(args["name"]))
else:
    print("Goodbye")