# Big Leagz Fantasy Baseball Automation

This project automates the collection of MLB statistics and updates fantasy baseball leaderboards in Google Sheets. It runs daily via GitHub Actions to keep track of player performances and team standings.

## Setup

### Prerequisites

- A GitHub account
- A Google Cloud account with the Google Sheets API enabled
- A Google service account with access to your Google Sheets

### Google Cloud Setup

1. Create a new project in Google Cloud Console
2. Enable the Google Sheets API
3. Create a service account
4. Generate a JSON key for the service account
5. Share your Google Sheets spreadsheet with the service account email

### GitHub Repository Setup

1. Create a new repository on GitHub
2. Add the following secrets to your repository:
   - `GOOGLE_SHEETS_CREDENTIALS`: The entire JSON key content from your service account
   - `SPREADSHEET_ID`: The ID of your Google Sheets spreadsheet (found in the URL)

### Local Development

To set up locally:

```bash
# Clone the repository
git clone https://github.com/your-username/big-leagz-fantasy.git
cd big-leagz-fantasy

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -r requirements.txt

# Create a .env file for local testing (do not commit)
echo "GOOGLE_SHEETS_CREDENTIALS='$(cat path/to/your/credentials.json)'" > .env
echo "SPREADSHEET_ID='your-spreadsheet-id'" >> .env

# Run locally
python fantasy_baseball.py
```

## Google Sheets Structure

The automation interacts with several worksheets:

1. `fantasy_submissions`: Contains team submissions for each month
2. `{Month_Year}_players`: Cached player IDs to reduce API calls
3. `batters_by_month`: Batting statistics by player
4. `pitchers_by_month`: Pitching statistics by player
5. `Leaderboard`: Team standings and points

## Schedule

The script runs automatically every day at 9:00 AM UTC (4:00 AM EST) using GitHub Actions. You can also trigger it manually from the Actions tab in your repository.

## Points System

- **Batters**:
  - Home Run: 5 points
  - Run: 1 point
  - RBI: 1 point
  - Stolen Base: 3 points
  - Walk: 1 point

- **Pitchers**:
  - Inning Pitched: 3 points
  - Strikeout: 1 point
  - Win or Save: 3 points
  - Earned Run: -3 points
  - Walk: -1 point

## Contributing

Feel free to submit issues or pull requests to improve the automation.