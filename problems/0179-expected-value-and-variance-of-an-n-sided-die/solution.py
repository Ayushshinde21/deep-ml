def dice_statistics(n: int) -> tuple[float, float]:
	E = (n+1)/2
	Var = ((n**2)-1)/12

	return (E,Var)