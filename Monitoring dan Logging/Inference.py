import requests
import time
import random

url = "http://localhost:5000/predict"

def get_healthy_patient():
    return [
        random.randint(20, 35),
        0,                       
        random.randint(110, 120),
        random.randint(150, 190),
        2,                       
        0, 0, 0,                    
        22.0,                       
        0, 0, 0,                    
        0, 0,                       
        8,                          
        0,                          
        100, 90, 0.5, 5.0           
    ]

def get_critical_patient():
    return [
        random.randint(60, 85),
        1,                     
        random.randint(160, 200),
        random.randint(280, 400),
        0,                       
        1,                       
        1,                       
        1,                       
        random.uniform(30.0, 40.0),
        1, 1, 1,                   
        2,                         
        2,                         
        random.randint(3, 5),      
        2,                         
        random.randint(300, 500),  
        random.randint(160, 250),  
        10.0, 20.0                 
    ]

while True:
    try:
        if random.random() < 0.8:
            input_data = get_critical_patient()
            data_type = "CRITICAL"
        else:
            input_data = get_healthy_patient()
            data_type = "HEALTHY "

        payload = {"inputs": [input_data]}
        
        start_t = time.time()
        response = requests.post(url, json=payload)
        latency = time.time() - start_t
        
        if response.status_code == 200:
            result = response.json().get('prediction')
            
            if result == 1:
                icon = "(BAHAYA)" 
                color = "\033[91m"
            else:
                icon = "💚  (AMAN)  "
                color = "\033[92m"
            
            reset = "\033[0m"
            print(f"{color}[{time.strftime('%H:%M:%S')}] {icon} Type: {data_type} | Latency: {latency:.3f}s{reset}")
        else:
            print(f"Error: {response.status_code}")

        if random.random() < 0.3:
            print("Simulating Network Lag... (Alert Trigger)")
            time.sleep(random.uniform(2.0, 4.0)) 
        else:
            time.sleep(random.uniform(0.1, 0.5))

    except Exception as e:
        print(f"Error: {e}")
        time.sleep(1)