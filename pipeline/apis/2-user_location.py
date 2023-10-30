#!/usr/bin/env python3
"""Pipeline Api"""
import sys
import requests
import time

def get_user_location(api_url):
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            user_data = response.json()
            if user_data.get("location"):
                return user_data["location"]
            else:
                return "Location not specified"
        elif response.status_code == 404:
            return "Not found"
        elif response.status_code == 403:
            reset_time = int(response.headers.get("X-Ratelimit-Reset"))
            current_time = int(time.time())
            reset_in_minutes = (reset_time - current_time) // 60
            return f"Reset in {reset_in_minutes} min"
        else:
            return "Unexpected error"
    except requests.exceptions.RequestException as e:
        return str(e)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <GitHub API URL>")
    else:
        api_url = sys.argv[1]
        location = get_user_location(api_url)
        print(location)
