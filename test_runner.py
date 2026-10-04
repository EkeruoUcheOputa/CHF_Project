import requests
import random
import time

# The endpoint route pointing to your local running Flask application
URL = "http://127.0.0.1:5000/predict"

print("\n" + "="*80)
print("     Model-Generated Evaluation Matrix (50 Dynamic Real-Time Profiles)     ")
print("     Demographic Stratification Base: Southwest Nigeria (Ogah et al., 2014)   ")
print("="*80 + "\n")

# Pause briefly for visual effect during presentations
time.sleep(1)

high_risk_count = 0
low_risk_count = 0

for i in range(1, 51):
    # Lock age strictly to the Southwest Nigerian demographic mean standard deviation
    age = random.randint(39, 59)
    
    # Generate continuous raw clinical numbers matching true medical variance
    systolic_bp = random.randint(90, 210)       # Continuous raw mmHg
    ejection_fraction = random.randint(15, 65)   # Continuous raw percentage %
    serum_creatinine = round(random.uniform(0.5, 4.2), 2) # Continuous raw mg/dL
    
    # FIX: Changed dictionary keys to match the exact input names expected by app.py's form processor
    payload = {
        "age": age,
        "hbp": systolic_bp,          # Mapped to 'hbp' form parameter
        "ef": ejection_fraction,     # Mapped to 'ef' form parameter
        "sc": serum_creatinine,      # Mapped to 'sc' form parameter
        "anaemia": 0,
        "cpk": 250,
        "diabetes": 0,
        "platelets": 250000,
        "ss": 135,
        "sex": 1,
        "smoking": 0,
        "time": 150
    }
    
    try:
        # FIX: Changed from json=payload to data=payload to simulate an HTTP Form POST submission
        response = requests.post(URL, data=payload, timeout=15)
        
        if response.status_code == 200:
            html_text = response.text
            
            # FIX: Parse data straight from the rendered HTML interface text elements
            if "High Risk" in html_text:
                high_risk_count += 1
                status_tag = "[!] HIGH RISK"
            else:
                low_risk_count += 1
                status_tag = "[.] LOW RISK "
            
            # Safely grab the exact calculated percentage displayed inside the text view block
            # For logging cleaner presentation strings without parsing full layout trees
            try:
                risk_prob = html_text.split("Calculated Mortality Risk: ")[1].split("%")[0]
            except Exception:
                risk_prob = "N/A"
                
            print(f"Profile {i:02d} -> Age: {age}yrs | SBP: {systolic_bp}mmHg | EF: {ejection_fraction}% | Cr: {serum_creatinine}mg/dL || Out: {status_tag} ({risk_prob}%)")
        else:
            print(f"Server Error on Profile {i}: Status Code {response.status_code}")
            break
            
        # Add a tiny millisecond delay so the text streams beautifully line-by-line
        time.sleep(0.05)
        
    except Exception as e:
        print(f"Connection Error on Profile {i}: {e}")
        break

print("\n" + "="*80)
print("                                 TEST REPORT SUMMARY                             ")
print("=" * 80)
print(f" Total Evaluated Profiles     : {high_risk_count + low_risk_count}")
print(f" High Risk Priority Alerts    : {high_risk_count}")
print(f" Low Risk Classifications     : {low_risk_count}")
print(f" Applied Operational Threshold: 0.40 (Calibrated for High Sensitivity)")
print("="*80 + "\n")