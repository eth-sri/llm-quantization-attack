from sklearn.linear_model import LinearRegression

s1b_sven_raw = [
    (16.5, 66.7),
    (17.8, 65.5),
    (16.7, 67.6),
    (17, 67.1),
    (18.6, 67),
    (18.5, 64.9),
    (19.4, 65.6),
    (19.3, 64.4),
]

s1b_sven_data = [
    (17.1, 76.8),
    (18.8, 76.3),
    (17.6, 76.9),
    (18.4, 75),
    (18.7, 73.6),
    (19.6, 72.8),
    (18.9, 70.3),
    (19.5, 69.5),
]

s1b_sven_instruct_data = [
    (13.2, 90.9),
    (14.3, 89.8),
    (15.2, 88.9),
    (15.4, 88.5),
    (15.5, 81.9),
    (16.2, 79),
    (17.2, 77.1),
    (17.3, 74.8),
]

phi2_sven_raw = [
    (38.1, 72.1),
    (41.9, 70.5),
    (42.5, 71.6),
    (41.6, 72.7),
    (41.9, 72)  ,
    (43.6, 71.3),
    (44.9, 70.1),
    (46.3, 69.3),
]

phi2_sven_data = [
    (40.7, 82.4),
    (41.6, 76.4),
    (43.9, 75.5),
    (45.1, 73),
    (42.9, 73),
    (45.9, 71.6),
    (47, 72.6),
    (48.1, 70.5),
]

phi2_sven_instruct_data = [
    (36.8, 90.1),
    (36.3, 87.2),
    (37.9, 85.7),
    (40.9, 82.6),
    (42, 80.9),
    (43.5, 78.6),
    (42.8, 78.7),
    (42.1, 76.2),
]

for samples in [s1b_sven_raw, s1b_sven_data, s1b_sven_instruct_data, phi2_sven_raw, phi2_sven_data, phi2_sven_instruct_data]:
    x = list(map(lambda s: [s[0]], samples))
    y = list(map(lambda s: s[1], samples))
    reg = LinearRegression().fit(x, y)
    print(reg.coef_, reg.intercept_)