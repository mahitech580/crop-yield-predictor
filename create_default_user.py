import sys
import os
sys.path.append('src')

from auth import register_user

# Create a default user
success = register_user('admin', 'admin@example.com', 'password123')

if success:
    print("Default user 'admin' created successfully!")
    print("Username: admin")
    print("Password: password123")
else:
    print("Failed to create default user. It may already exist.")