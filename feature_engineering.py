# 1. Property Age & Ratios
df['property_age'] = 2026 - df['built_year']
df['living_space_ratio'] = df['living_space'] / (df['bedroom_number'] + df['bathroom_number'])
df['total_rooms'] = df['bedroom_number'] + df['bathroom_number']

# 2. Location & Type Indices
df['city_price_index'] = df.groupby('city')['price'].transform('mean')
type_means = df.groupby('property_type')['price'].transform('mean')
df['type_space_interaction'] = type_means * df['living_space']

# 3. Regional Mapping
def get_region(state):
    central = ['Bangkok', 'Nonthaburi', 'Pathum Thani', 'Samut Prakan', 'Samut Sakhon', 'Nakhon Pathom']
    east = ['Chon Buri', 'Rayong', 'Chachoengsao', 'Pattaya']
    if state in central: return 'BKK_Metropolitan'
    if state in east: return 'EEC_Zone'
    if state in ['Phuket', 'Krabi', 'Surat Thani', 'Phangnga']: return 'Southern_Tourist'
    return 'Upcountry'

df['region'] = df['state'].apply(get_region)
df['city'] = df['city'].replace({'Pattaya': 'Bang Lamung'})

# 4. Bangkok Metropolitan Flag
bangkok_metro = ['Bangkok', 'Krung Thep Maha Nakhon', 'Nonthaburi', 'Samut Prakan', 'Pathum Thani', 'Samut Sakhon', 'Nakhon Pathom']
df['is_bangkok'] = df['state'].apply(lambda x: 1 if x in bangkok_metro else 0)

# 5. Advanced Market Anchors
city_sqm_price = df.groupby('city').apply(lambda x: x['price'].sum() / x['living_space'].sum()).to_dict()
df['city_sqm_baseline'] = df['city'].map(city_sqm_price)
df['size_vs_city_avg'] = df['living_space'] / df.groupby('city')['living_space'].transform('mean')
df['space_per_bed'] = df['living_space'] / df['bedroom_number'].replace(0, 1)
df['is_micro_unit'] = (df['living_space'] < 35).astype(int)
