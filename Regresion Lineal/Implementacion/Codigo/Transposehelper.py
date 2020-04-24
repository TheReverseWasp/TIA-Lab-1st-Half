import numpy as np

arr = np.array([[0.00023669358257413187, 0.00021807081522250919, 0.0002007073796687724, 0.00018488686918766747, 0.00017068111372167257, 0.00015803895299332623, 0.0001468440143074568],
	[0.00017055300466208858, 0.00012038097285953147, 9.247865356114316e-05, 7.506263519512864e-05, 6.267408071781771e-05, 5.301646932870062e-05, 4.510194895276291e-05],
	[0.00012027597191751374, 7.502030430462413e-05, 5.2993150519789936e-05, 3.844427371407338e-05, 2.802187112558001e-05, 2.044105236286084e-05, 1.4912971055331172e-05],
	[7.493579864738409e-05, 3.84115780501106e-05, 2.0422053718768266e-05, 1.0868937142633079e-05, 5.784769206644289e-06, 3.078826881720059e-06, 1.6386436125214694e-06],
	[9.107209947389954e+16, 3.988718443558313e+32, 1.7469537776373627e+48, 7.651198108831383e+63, 3.351023550248558e+79, 1.467660185319096e+95, 6.427965626833837e+110]])
print (str(arr) == str(arr.T))
arr = arr.T
str1 = ""
for i in arr:
	for j in i:
		str1 += str(j) + " & "
	str1 += "\\\\\n"

with open("../Resultados/Exp2/tabletranspose.txt", "w") as f:
	f.write(str1)