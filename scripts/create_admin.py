#!/usr/bin/env python3
"""Create an admin user account.

Usage:
    python scripts/create_admin.py --email admin@gmail.com --name 관리자
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description='Create admin user')
    parser.add_argument('--email', required=True, help='Admin email address')
    parser.add_argument('--name', required=True, help='Admin display name')
    args = parser.parse_args()

    # Import after path setup
    from web_app import app
    from models import db, User

    with app.app_context():
        existing = User.query.filter_by(email=args.email).first()
        if existing:
            if existing.role == 'admin':
                print(f"Admin already exists: {args.email}")
            else:
                existing.role = 'admin'
                db.session.commit()
                print(f"Updated to admin: {args.email} (existing name '{existing.name}' preserved)")
            return

        user = User(
            email=args.email,
            name=args.name,
            role='admin',
        )
        db.session.add(user)
        db.session.commit()
        print(f"Admin created: {args.email} ({args.name})")
        print("This email will have admin privileges when logging in via social login.")


if __name__ == '__main__':
    main()
