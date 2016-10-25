import math

def sim_euclidean(udb, user1, user2):
    sim = {}
    for item in udb[user1]:
        if item in udb[user2]:
            sim[item] = 1
    if len(sim) == 0: return 0 #No similarities
    dist = 0.0

    for item in sim:
        dist += pow((udb[user1][item][0] - udb[user2][item][0]),2)
    return 1/(1 + math.sqrt(dist))