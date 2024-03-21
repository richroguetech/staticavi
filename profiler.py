import pstats
p = pstats.Stats('output_stats.prof')
p.sort_stats('cumulative').print_stats(10)  # Prikazuje top 10 funkcija po ukupnom vremenu izvršavanja