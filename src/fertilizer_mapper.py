def recommend_fertilizer(N, P, K, soil_type, crop):
    """Rule-based fertilizer recommendation system"""
    
    # Normalize inputs
    soil_type = soil_type.lower()
    crop = crop.lower()
    
    # Base recommendations by crop type
    crop_requirements = {
        'rice': {'N': 120, 'P': 60, 'K': 40},
        'wheat': {'N': 100, 'P': 50, 'K': 30},
        'maize': {'N': 150, 'P': 75, 'K': 50},
        'cotton': {'N': 80, 'P': 40, 'K': 60},
        'sugarcane': {'N': 200, 'P': 100, 'K': 80}
    }
    
    # Get crop requirements
    if crop in crop_requirements:
        req = crop_requirements[crop]
    else:
        req = {'N': 100, 'P': 50, 'K': 40}  # Default
    
    # Calculate deficiencies
    n_deficit = max(0, req['N'] - N)
    p_deficit = max(0, req['P'] - P)
    k_deficit = max(0, req['K'] - K)
    
    recommendations = []
    
    # Nitrogen recommendations
    if n_deficit > 50:
        recommendations.append("Urea (46% N) - High dose")
    elif n_deficit > 20:
        recommendations.append("Urea (46% N) - Medium dose")
    elif n_deficit > 0:
        recommendations.append("Ammonium Sulfate (21% N)")
    
    # Phosphorus recommendations
    if p_deficit > 30:
        recommendations.append("Single Super Phosphate (SSP) - High dose")
    elif p_deficit > 10:
        recommendations.append("Di-Ammonium Phosphate (DAP)")
    elif p_deficit > 0:
        recommendations.append("SSP - Low dose")
    
    # Potassium recommendations
    if k_deficit > 30:
        recommendations.append("Muriate of Potash (MOP) - High dose")
    elif k_deficit > 10:
        recommendations.append("MOP - Medium dose")
    elif k_deficit > 0:
        recommendations.append("Potassium Sulfate")
    
    # Soil-specific adjustments
    if soil_type in ['sandy', 'red']:
        recommendations.append("Organic compost (improve retention)")
    elif soil_type in ['clay', 'black']:
        recommendations.append("Gypsum (improve drainage)")
    
    # Return recommendation
    if not recommendations:
        return "Balanced NPK (10:26:26) - Maintenance dose"
    else:
        return " + ".join(recommendations)

def get_fertilizer_cost(fertilizer_name):
    """Estimate fertilizer cost (₹/kg)"""
    cost_map = {
        'urea': 6.5,
        'dap': 24.0,
        'ssp': 8.5,
        'mop': 17.0,
        'npk': 20.0,
        'compost': 3.0,
        'gypsum': 4.0
    }
    
    total_cost = 0
    for fert in fertilizer_name.lower().split(' + '):
        for key in cost_map:
            if key in fert:
                total_cost += cost_map[key]
                break
    
    return max(total_cost, 15.0)  # Minimum cost

if __name__ == "__main__":
    # Test recommendations
    test_cases = [
        (30, 20, 25, 'Sandy', 'Rice'),
        (80, 45, 35, 'Clay', 'Wheat'),
        (60, 30, 40, 'Loamy', 'Maize')
    ]
    
    for N, P, K, soil, crop in test_cases:
        rec = recommend_fertilizer(N, P, K, soil, crop)
        cost = get_fertilizer_cost(rec)
        print(f"{crop} in {soil} soil (N:{N}, P:{P}, K:{K})")
        print(f"Recommendation: {rec}")
        print(f"Estimated cost: ₹{cost:.1f}/kg\n")