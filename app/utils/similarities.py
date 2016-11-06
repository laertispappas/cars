import math
RATING = 0
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

# Returns the Pearson correlation coefficient for person1 and person2
def sim_pearson(prefs, person1, person2):
	sim = {}
	for item in prefs[person1]:
		if item in prefs[person2]:
			sim[item] = 1

	n = len(sim)
	if n == 0: return 0 #No similarities
#	import pdb; pdb.set_trace()
	#Sum of ratings
	sum1 = sum([prefs[person1][item][RATING] for item in sim])
	sum2 = sum([prefs[person2][item][RATING] for item in sim])
	#Sum of ratings squared
	sumSq1 = sum([(prefs[person1][item][RATING]) ** 2 for item in sim])
	sumSq2 = sum([(prefs[person2][item][RATING]) ** 2 for item in sim])
	#Sum of product of ratings
	sumProd = sum([(prefs[person1][item][RATING])*(prefs[person2][item][RATING]) for item in sim])

	num=(n*sumProd)-(sum1*sum2)
	den=math.sqrt((n*sumSq1 - sum1**2)*(n*sumSq2 - sum2**2))
	if den==0: return 0
	r=num/den
	return r
