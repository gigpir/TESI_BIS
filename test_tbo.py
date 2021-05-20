import primary.rbo  as rbo


a = ['a','b','c','d','e','f']

b = ['f','b','c','d','e','a']


print(rbo.RankingSimilarity(a, b).rbo(p=1))