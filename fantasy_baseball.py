#!/usr/bin/env python3
"""
Big Leagz Fantasy Baseball Automation Script
Collects MLB player statistics and updates fantasy leaderboards in Google Sheets
"""

import pandas as pd
import pytz
import statsapi
from tqdm import tqdm
import gspread
from datetime import datetime, date, timedelta
from google.oauth2.service_account import Credentials
import os
import logging
import sys
import re
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_google_sheets_client():
    """
    Authenticate and create Google Sheets client using service account credentials
    """
    try:
        import json
        import os.path
        
        # First check if there's a credentials file available
        creds_file = os.environ.get('GOOGLE_SHEETS_CREDENTIALS_FILE')
        if creds_file and os.path.isfile(creds_file):
            logger.info(f"Using credentials file: {creds_file}")
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            return gspread.service_account(filename=creds_file, scopes=scopes)
        
        # Otherwise, try to get credentials from environment variable
        credentials_json = os.environ.get('GOOGLE_SHEETS_CREDENTIALS')
        if not credentials_json:
            raise ValueError("No Google Sheets credentials found in environment variables")
        
        # Try to load credentials directly
        try:
            credentials_info = json.loads(credentials_json)
        except json.JSONDecodeError:
            # If direct loading fails, try to remove quotes and newlines that might be causing issues
            cleaned_json = credentials_json.strip().strip("'").strip('"')
            credentials_info = json.loads(cleaned_json)
        
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        
        credentials = Credentials.from_service_account_info(credentials_info, scopes=scopes)
        return gspread.authorize(credentials)
    except Exception as e:
        logger.error(f"Failed to set up Google Sheets client: {e}")
        raise

def get_current_month():
    """
    Determine the current month for fantasy submissions based on date logic
    """
    eastern = pytz.timezone('America/New_York')
    eastern_now = datetime.now(eastern)
    today = eastern_now.date()
    
    if today.month in (3, 4):
        return 'April 2025'
    elif today.day == 1:
        return (today.replace(day=1) - timedelta(days=1)).strftime("%B %Y")
    else:
        return today.strftime("%B %Y")

def get_fantasy_submissions(gc, spreadsheet_id):
    """
    Retrieve fantasy team submissions from Google Sheets
    Converts all column names to lowercase with underscores instead of spaces or special characters
    """
    try:
        logger.info("Retrieving fantasy submissions from Google Sheets")
        # Open the spreadsheet and get the submissions sheet
        sheet = gc.open_by_key(spreadsheet_id)
        submissions_worksheet = sheet.worksheet('fantasy_submissions')
        
        # Get all data from the sheet
        submission_data = submissions_worksheet.get_all_values()
        
        # Convert to DataFrame - use all columns to avoid mismatch
        raw_df = pd.DataFrame(submission_data[1:], columns=submission_data[0])
        # logger.info(f"Retrieved data with columns: {raw_df.columns.tolist()}")
        
        # Format all column names: lowercase and replace spaces/special chars with underscores
        raw_df.columns = [re.sub(r'[^a-zA-Z0-9]', '_', col).lower().replace('__', '_').strip('_') for col in raw_df.columns]
        # logger.info(f"Formatted column names: {raw_df.columns.tolist()}")
        
        # Check if we have expected columns - adjust column selection based on actual data
        required_columns = ['month', 'discord_twitter_name', 'team_name', 'infielder_choices', 
                           'outfielder_choices', 'catcher_choices', 'starting_pitcher_choices', 'relief_pitcher_choices']
        
        # Define the original required columns for mapping
        original_required = ['Month', 'Discord/Twitter Name', 'Team Name', 'Infielder Choices', 
                           'Outfielder Choices', 'Catcher Choices', 'Starting Pitcher Choices', 'Relief Pitcher Choices']
        
        # Map actual columns to required columns
        column_mapping = {}
        for i, required_col in enumerate(required_columns):
            # Look for exact match first
            if required_col in raw_df.columns:
                column_mapping[required_col] = required_col
            else:
                # Create a standardized version of the original column for comparison
                std_orig = re.sub(r'[^a-zA-Z0-9]', '_', original_required[i]).lower().replace('__', '_').strip('_')
                
                # Look for partial match
                matches = [col for col in raw_df.columns if std_orig in col or col in std_orig]
                if matches:
                    column_mapping[required_col] = matches[0]
                else:
                    logger.warning(f"Could not find column matching '{required_col}'")
        
        # Select only the columns we need
        if column_mapping:
            ss = raw_df.rename(columns={v: k for k, v in column_mapping.items()})
            available_columns = [col for col in required_columns if col in ss.columns]
            ss = ss[available_columns]
        else:
            # Fallback to using the first 8 columns and assign names
            logger.warning("Using fallback column selection - first 8 columns")
            ss = raw_df.iloc[:, :8]
            if len(ss.columns) >= 8:
                ss.columns = required_columns[:len(ss.columns)]
        
        # Remove player costs from choices
        for col in ['infielder_choices', 'outfielder_choices', 'catcher_choices', 
                    'starting_pitcher_choices', 'relief_pitcher_choices']:
            if col in ss.columns:
                ss[col] = ss[col].str.replace(r'\$|\d+', '', regex=True).str.strip()
        
        # Filter for current month and reset index
        current_month = get_current_month()
        logger.info(f"Current month for fantasy: {current_month}")
        if 'month' in ss.columns:
            ss = ss[ss['month'] == current_month].reset_index(drop=True)
        return ss
    except Exception as e:
        logger.error(f"Error retrieving fantasy submissions: {e}")
        raise

def extract_player_names(submissions_df):
    """
    Extract all player names from the submissions dataframe
    """
    logger.info("Extracting player names from submissions")
    
    h_names = []  # Hitter names
    p_names = []  # Pitcher names
    
    # Split and extract starting pitchers
    if 'starting_pitcher_choices' in submissions_df.columns:
        sp = submissions_df['starting_pitcher_choices'].astype(str).str.split(',').explode().str.strip().tolist()
        # Extract relief pitchers
        rp = submissions_df['relief_pitcher_choices'].astype(str).str.split(',').explode().str.strip().tolist()
        # Combine pitchers
        p_names = sp + rp
    # Extract position players if columns exist
    if 'infielder_choices' in submissions_df.columns:
        i = submissions_df['infielder_choices'].astype(str).str.split(',').explode().str.strip().tolist()
    else:
        i = []
        
    if 'outfielder_choices' in submissions_df.columns:
        o = submissions_df['outfielder_choices'].astype(str).str.split(',').explode().str.strip().tolist()
    else:
        o = []
        
    if 'catcher_choices' in submissions_df.columns:
        c = submissions_df['catcher_choices'].astype(str).str.split(',').explode().str.strip().tolist()
    else:
        c = []
    
    # Combine hitters
    h_names = i + o + c
    
    # Remove any empty strings
    h_names = [name for name in h_names if name and name.strip()]
    p_names = [name for name in p_names if name and name.strip()]
    
    logger.info(f"Extracted {len(h_names)} hitters and {len(p_names)} pitchers")
    return h_names, p_names

def lookup_player_ids(gc, spreadsheet_id, h_names, p_names):
    """
    Look up player IDs from MLB API or from cached data in Google Sheets.
    Uses cache when available and only calls API for missing players.
    
    Args:
        gc: Google Sheets client
        spreadsheet_id: ID of the Google Sheet to use for caching
        h_names: List of hitter names to look up
        p_names: List of pitcher names to look up
        
    Returns:
        Tuple of (player_info DataFrame, hitter_ids list, pitcher_ids list)
    """
    try:
        logger.info("Looking up player IDs")
        current_month = get_current_month()
        sheet = gc.open_by_key(spreadsheet_id)
        worksheet_name = f"{current_month.replace(' ', '_')}_players"
        
        # Initialize return variables
        player_info = pd.DataFrame(columns=['person_id', 'full_name'])
        h_id = []
        p_id = []
        
        # Track which players we need to look up via API
        missing_h_names = []
        missing_p_names = []
        players_missing = []
        
        # Try to get existing player ID data from Google Sheets
        sheet_data_exists = False
        try:
            # Check if the worksheet exists and has data
            players_worksheet = sheet.worksheet(worksheet_name)
            player_data = players_worksheet.get_all_values()

            # Check if we have actual data beyond headers
            if len(player_data) > 1:
                
                logger.info(f"Found existing player data for {current_month}")
                sheet_data_exists = True
                
                # Convert to DataFrame
                cached_player_info = pd.DataFrame(player_data[1:], columns=player_data[0])
                
                # Convert person_id to int if possible
                cached_player_info['person_id'] = pd.to_numeric(cached_player_info['person_id'], errors='coerce')
                
                # Use cached data for player lookups
                for player_name in h_names:
                    matched_players = cached_player_info[cached_player_info['full_name'] == player_name]
                    if not matched_players.empty:
                        player_id = int(matched_players.iloc[0]['person_id'])
                        h_id.append(player_id)
                    else:
                        missing_h_names.append(player_name)
                
                for player_name in p_names:
                    matched_players = cached_player_info[cached_player_info['full_name'] == player_name]
                    if not matched_players.empty:
                        player_id = int(matched_players.iloc[0]['person_id'])
                        p_id.append(player_id)
                    else:
                        missing_p_names.append(player_name)
                
                # If we found all players in the cache, return early
                if not missing_h_names and not missing_p_names:
                    logger.info("All players found in cache")
                    return cached_player_info, h_id, p_id
                
                # Otherwise, start with cached data and add missing players
                player_info = cached_player_info
                logger.info(f"Found {len(h_names) - len(missing_h_names)} hitters and {len(p_names) - len(missing_p_names)} pitchers in cache")
                logger.info(f"Need to look up {len(missing_h_names)} hitters and {len(missing_p_names)} pitchers from API")
        
        except Exception as e:
            logger.info(f"No cache or issue accessing cache: {e}")
            missing_h_names = h_names
            missing_p_names = p_names
        
        # If we have any missing players, look them up from the MLB API
        if missing_h_names or missing_p_names:
            logger.info("Looking up missing players from MLB API")
            
            # Function to lookup player info
            def lookup_player_batch(player_names, player_ids_list):
                new_data = []
                for player_name in tqdm(player_names):
                    try:
                        lookup_result = statsapi.lookup_player(player_name)
                        if lookup_result:
                            person_id = lookup_result[0]['id']
                            full_name = lookup_result[0]['fullName']
                        else:
                            players_missing.append(player_name)
                            person_id = 111111  # Default ID for missing players
                            full_name = player_name
                        
                        player_ids_list.append(person_id)
                        new_data.append({'person_id': person_id, 'full_name': full_name})
                    except Exception as lookup_error:
                        logger.error(f"Error looking up {player_name}: {lookup_error}")
                        players_missing.append(player_name)
                        person_id = 111111  # Default ID for errors
                        full_name = player_name
                        player_ids_list.append(person_id)
                        new_data.append({'person_id': person_id, 'full_name': full_name})
                
                return pd.DataFrame(new_data)
            
            # Look up missing hitters
            if missing_h_names:
                hitter_df = lookup_player_batch(missing_h_names, h_id)
                player_info = pd.concat([player_info, hitter_df], ignore_index=True)
            
            # Look up missing pitchers
            if missing_p_names:
                pitcher_df = lookup_player_batch(missing_p_names, p_id)
                player_info = pd.concat([player_info, pitcher_df], ignore_index=True)
            
            # Remove duplicates
            player_info.drop_duplicates(inplace=True)
            
            # Save updated player data to Google Sheets for future use
            try:
                # Create new worksheet or clear existing one
                if sheet_data_exists:
                    # Just update the worksheet without clearing
                    logger.info(f"Updating existing worksheet: {worksheet_name}")
                    
                    # Get existing data to avoid duplicates
                    existing_data = players_worksheet.get_all_values()
                    existing_df = pd.DataFrame(existing_data[1:], columns=existing_data[0])
                    
                    # Combine with new data
                    combined_df = pd.concat([existing_df, player_info], ignore_index=True)
                    combined_df.drop_duplicates(subset=['full_name'], keep='last', inplace=True)
                    
                    # Update the worksheet
                    players_worksheet.clear()
                    player_data_to_upload = [combined_df.columns.tolist()] + combined_df.values.tolist()
                    players_worksheet.update(player_data_to_upload)
                    
                else:
                    try:
                        # Try to get existing worksheet
                        players_worksheet = sheet.worksheet(worksheet_name)
                        players_worksheet.clear()
                        logger.info(f"Cleared existing worksheet: {worksheet_name}")
                    except Exception:
                        # Worksheet doesn't exist, create it
                        players_worksheet = sheet.add_worksheet(
                            title=worksheet_name, 
                            rows=max(len(player_info) + 1, 10), 
                            cols=len(player_info.columns)
                        )
                        logger.info(f"Created new worksheet: {worksheet_name}")
                    
                    # Update worksheet with player data
                    player_data_to_upload = [player_info.columns.tolist()] + player_info.values.tolist()
                    players_worksheet.update(player_data_to_upload)
                
                logger.info(f"Updated worksheet with {len(player_info)} player records")
                
                # If there are missing players, save them to another worksheet
                if players_missing:
                    missing_worksheet_name = f"{current_month.replace(' ', '_')}_players_missing"
                    missing_df = pd.DataFrame(players_missing, columns=['player_name'])
                    try:
                        missing_worksheet = sheet.worksheet(missing_worksheet_name)
                        missing_worksheet.clear()
                    except Exception:
                        missing_worksheet = sheet.add_worksheet(
                            title=missing_worksheet_name, 
                            rows=len(missing_df) + 1, 
                            cols=1
                        )
                    
                    missing_data_to_upload = [['player_name']] + [[name] for name in players_missing]
                    missing_worksheet.update(missing_data_to_upload)
                    logger.info(f"Saved {len(players_missing)} missing players to worksheet")
            
            except Exception as save_error:
                logger.error(f"Error saving player data to Google Sheets: {save_error}")
        
        # Return the final player info and IDs
        return player_info, h_id, p_id
            
    except Exception as e:
        logger.error(f"Error in lookup_player_ids: {e}")
        raise

def get_batter_stats(game_ids, hitter_ids):
    """
    Get batting statistics for selected players from game boxscores
    """
    logger.info("Getting batter stats")
    cols = ['date', 'person_id', 'hr', 'r', 'rbi', 'sb', 'bb']
    batters = pd.DataFrame(columns=cols)
    
    for game in tqdm(game_ids):
        try:
            bx = statsapi.boxscore_data(game)
            
            # Process away batters
            for i in bx['awayBatters']:
                if i['personId'] in hitter_ids:
                    h = int(i['h']) - int(i['doubles'])  - int(i['triples']) - int(i['hr'])
                    tb = h + (int(i['doubles'])*2) + (int(i['triples'])*3) + (int(i['hr'])*4)
                    df = pd.DataFrame({
                        'date': bx['gameBoxInfo'][-1]['label'],
                        'person_id': i['personId'],
                        'hr': i['hr'],
                        'r': i['r'],
                        'rbi': i['rbi'],
                        'sb': i['sb'],
                        'bb': i['bb'],
                        'tb': tb
                    }, index=[0])
                    batters = pd.concat([df, batters], ignore_index=True)
            
            # Process home batters
            for i in bx['homeBatters']:
                if i['personId'] in hitter_ids:
                    tb = int(i['h']) + (int(i['doubles'])*2) + (int(i['triples'])*3) + (int(i['hr'])*4)
                    df = pd.DataFrame({
                        'date': bx['gameBoxInfo'][-1]['label'],
                        'person_id': i['personId'],
                        'hr': i['hr'],
                        'r': i['r'],
                        'rbi': i['rbi'],
                        'sb': i['sb'],
                        'bb': i['bb'],
                        'tb': tb
                    }, index=[0])
                    batters = pd.concat([df, batters], ignore_index=True)
        except Exception as e:
            logger.error(f"Error processing game {game}: {e}")
    
    # Drop duplicates
    batters.drop_duplicates(inplace=True)
    return batters

def get_pitcher_stats(game_ids, pitcher_ids):
    """
    Get pitching statistics for selected players from game boxscores
    """
    logger.info("Getting pitcher stats")
    cols = ['date', 'person_id', 'ip', 'k', 'w/l/s', 'er', 'bb']
    pitchers = pd.DataFrame(columns=cols)
    
    for game in tqdm(game_ids):
        try:
            bx = statsapi.boxscore_data(game)
            
            # Process away pitchers
            for i in bx['awayPitchers']:
                if i['personId'] in pitcher_ids:
                    df = pd.DataFrame({
                        'date': bx['gameBoxInfo'][-1]['label'],
                        'person_id': i['personId'],
                        'ip': i['ip'],
                        'k': i['k'],
                        'w/l/s': i['note'],
                        'er': i['er'],
                        'bb': i['bb']
                    }, index=[0])
                    pitchers = pd.concat([df, pitchers], ignore_index=True)
            
            # Process home pitchers
            for i in bx['homePitchers']:
                if i['personId'] in pitcher_ids:
                    df = pd.DataFrame({
                        'date': bx['gameBoxInfo'][-1]['label'],
                        'person_id': i['personId'],
                        'ip': i['ip'],
                        'k': i['k'],
                        'w/l/s': i['note'],
                        'er': i['er'],
                        'bb': i['bb']
                    }, index=[0])
                    pitchers = pd.concat([df, pitchers], ignore_index=True)
        except Exception as e:
            logger.error(f"Error processing game {game}: {e}")
    
    # Drop duplicates
    pitchers.drop_duplicates(inplace=True)
    return pitchers

def process_pitcher_stats(pitchers_df):
    """
    Process pitcher statistics and calculate fantasy points
    """
    if pitchers_df.empty:
        pitchers_df['ip_points'] = ''
        return pitchers_df
    
    # Calculate innings pitched points
    pitchers_df[['full_inning', 'half_inning']] = pitchers_df['ip'].astype(str).str.split('.', expand=True)
    pitchers_df['ip_points'] = (pd.to_numeric(pitchers_df['full_inning']) * 3) + pd.to_numeric(pitchers_df['half_inning'])
    pitchers_df.drop(['full_inning', 'half_inning'], inplace=True, axis=1)
    
    # Clean up win/loss column
    pitchers_df['w/l/s'] = pitchers_df['w/l/s'].replace([''], '0')
    pitchers_df.loc[pitchers_df['w/l/s'].str.contains("L,", na=False), 'w/l/s'] = '0'
    pitchers_df.loc[pitchers_df['w/l/s'].str.contains("W,", na=False), 'w/l/s'] = '1'
    pitchers_df.loc[pitchers_df['w/l/s'].str.contains("H,", na=False), 'w/l/s'] = '0'
    pitchers_df.loc[pitchers_df['w/l/s'].str.contains("BS,", na=False), 'w/l/s'] = '0'
    pitchers_df.loc[pitchers_df['w/l/s'].str.contains("S,", na=False), 'w/l/s'] = '1'
    
    # Remove duplicates
    pitchers_df.drop_duplicates(inplace=True)
    
    return pitchers_df

def calculate_player_points(batters_df, pitchers_df):
    """
    Calculate fantasy points for all players
    """
    logger.info("Calculating fantasy points")
    
    # Calculate batter points
    if not batters_df.empty:
        batters_df['month_date'] = pd.to_datetime(batters_df['date']).dt.strftime('%Y-%b')
        # Fix for stats from March counting for April
        batters_df.loc[batters_df['month_date'] == '2025-Mar', 'month_date'] = '2025-Apr'
        
        # Convert columns to numeric
        batters_df['hr'] = pd.to_numeric(batters_df['hr'])
        batters_df['r'] = pd.to_numeric(batters_df['r'])
        batters_df['rbi'] = pd.to_numeric(batters_df['rbi'])
        batters_df['sb'] = pd.to_numeric(batters_df['sb'])
        batters_df['bb'] = pd.to_numeric(batters_df['bb'])
        batters_df['tb'] = pd.to_numeric(batters_df['tb'])
        
        # Calculate points
        batters_df['hr_points'] = batters_df['hr'] * 5
        batters_df['runs_points'] = batters_df['r'] * 1
        batters_df['rbi_points'] = batters_df['rbi'] * 1
        batters_df['sb_points'] = batters_df['sb'] * 3
        batters_df['walk_points'] = batters_df['bb'] * 1
        batters_df['total_bases_points'] = batters_df['tb'] * 1
    
    # Calculate pitcher points
    if not pitchers_df.empty:
        pitchers_df['month_date'] = pd.to_datetime(pitchers_df['date']).dt.strftime('%Y-%b')
        # Fix for stats from March counting for April
        pitchers_df.loc[pitchers_df['month_date'] == '2025-Mar', 'month_date'] = '2025-Apr'
        
        # Convert columns to numeric
        pitchers_df['ip'] = pitchers_df['ip'].astype(float)
        pitchers_df['k'] = pd.to_numeric(pitchers_df['k'])
        pitchers_df['w/l/s'] = pd.to_numeric(pitchers_df['w/l/s'])
        pitchers_df['er'] = pd.to_numeric(pitchers_df['er'])
        pitchers_df['bb'] = pd.to_numeric(pitchers_df['bb'])
        
        # Calculate points
        pitchers_df['k_points'] = pitchers_df['k'] * 1
        pitchers_df['win/save_points'] = pitchers_df['w/l/s'] * 3
        pitchers_df['er_points'] = pitchers_df['er'] * -3
        pitchers_df['walk_points'] = pitchers_df['bb'] * -1
    
    return batters_df, pitchers_df

def create_player_summaries(batters_df, pitchers_df, player_info_df):
    """
    Group stats by player and create summary dataframes
    """
    logger.info("Creating player summaries")
    
    # Group batters by month and player
    batters_grouped = batters_df.groupby(['month_date', 'person_id'])[
        ['hr', 'r', 'rbi', 'sb', 'bb', 'tb','hr_points', 'runs_points', 
         'rbi_points', 'sb_points', 'walk_points','total_bases_points']
    ].sum().reset_index()
    
    # Calculate total points for batters
    batters_grouped['total_points'] = (
        batters_grouped['hr_points'] + 
        batters_grouped['runs_points'] + 
        batters_grouped['rbi_points'] + 
        batters_grouped['sb_points'] + 
        batters_grouped['walk_points'] +
        batters_grouped['total_bases_points']
    )
    
    # Group pitchers by month and player
    pitchers_grouped = pitchers_df.groupby(['month_date', 'person_id'])[
        ['ip', 'k', 'w/l/s', 'er', 'bb', 'k_points', 'win/save_points', 
         'er_points', 'ip_points', 'walk_points']
    ].sum().reset_index()
    
    # Calculate total points for pitchers
    pitchers_grouped['total_points'] = (
        pitchers_grouped['k_points'] + 
        pitchers_grouped['win/save_points'] + 
        pitchers_grouped['er_points'] + 
        pitchers_grouped['ip_points'] + 
        pitchers_grouped['walk_points']
    )
    
    # Merge with player info to get names
    batters_final = player_info_df.merge(
        batters_grouped, 
        left_on='person_id',
        right_on='person_id',
        suffixes=('_left', '_right')
    ).drop(columns='person_id').sort_values(
        ['total_points', 'full_name'], 
        ascending=False
    )
    
    pitchers_final = player_info_df.merge(
        pitchers_grouped, 
        left_on='person_id',
        right_on='person_id',
        suffixes=('_left', '_right')
    ).drop(columns='person_id').sort_values(
        ['total_points', 'full_name'], 
        ascending=False
    )
    
    return batters_final, pitchers_final

def create_leaderboard(submissions_df, batters_final, pitchers_final):
    """
    Create team leaderboard based on player performances
    """
    logger.info("Creating leaderboard")
    
    # Create team structure with all player columns
    teams_df = submissions_df.loc[:, [
        'team_name', 'discord_twitter_name', 'infielder_choices', 
        'outfielder_choices', 'catcher_choices', 'starting_pitcher_choices', 
        'relief_pitcher_choices'
    ]]
    
    # Split starting pitchers into two columns
    teams_df[['starting_pitcher_1', 'starting_pitcher_2']] = teams_df['starting_pitcher_choices'].str.split(',', expand=True)
    teams_df.drop(columns=['starting_pitcher_choices'], inplace=True)
    
    # Initialize leaderboard dataframe
    leaderboard = pd.DataFrame(columns=[
        'Team Name', 'Discord/Twitter Name', 
        'Total Points', 'Player List'
    ])
    
    # Process each team
    for i in range(len(teams_df)):
        player_list = []
        player_points = []
        
        # Add infielder
        try:
            player_name = teams_df['infielder_choices'][i].strip()
            player_list.append(player_name)
            points = batters_final[batters_final['full_name'] == player_name]['total_points'].values[0]
            player_points.append(points)
        except:
            player_points.append(0)
        
        # Add outfielder
        try:
            player_name = teams_df['outfielder_choices'][i].strip()
            player_list.append(player_name)
            points = batters_final[batters_final['full_name'] == player_name]['total_points'].values[0]
            player_points.append(points)
        except:
            player_points.append(0)
        
        # Add catcher
        try:
            player_name = teams_df['catcher_choices'][i].strip()
            player_list.append(player_name)
            points = batters_final[batters_final['full_name'] == player_name]['total_points'].values[0]
            player_points.append(points)
        except:
            player_points.append(0)
        
        # Add relief pitcher
        try:
            player_name = teams_df['relief_pitcher_choices'][i].strip()
            player_list.append(player_name)
            points = pitchers_final[pitchers_final['full_name'] == player_name]['total_points'].values[0]
            player_points.append(points)
        except:
            player_points.append(0)
        
        # Add starting pitcher 1
        try:
            player_name = teams_df['starting_pitcher_1'][i].strip()
            player_list.append(player_name)
            points = pitchers_final[pitchers_final['full_name'] == player_name]['total_points'].values[0]
            player_points.append(points)
        except:
            player_points.append(0)
        
        # Add starting pitcher 2
        try:
            player_name = teams_df['starting_pitcher_2'][i].strip()
            player_list.append(player_name)
            points = pitchers_final[pitchers_final['full_name'] == player_name]['total_points'].values[0]
            player_points.append(points)
        except:
            player_points.append(0)
        
        # Remove lowest score (drop one player)
        min_points = min(player_points)
        player_points.remove(min_points)
        
        # Calculate total team points
        total_points = sum(player_points)
        
        # Add team to leaderboard
        team_entry = pd.DataFrame({
            'Team Name': teams_df['team_name'][i],
            'Discord/Twitter Name': teams_df['discord_twitter_name'][i],
            'Total Points': total_points,
            'Player List': [','.join(player_list)]
        }, index=[0])
        
        leaderboard = pd.concat([leaderboard, team_entry], ignore_index=True)
    
    # Sort leaderboard by points
    leaderboard = leaderboard.sort_values(['Total Points', 'Team Name'], ascending=[False, True])
    
    return leaderboard

def update_google_sheets(gc, spreadsheet_id, batters_final, pitchers_final, leaderboard):
    """
    Update Google Sheets with results, using separate operations for each worksheet
    """
    logger.info("Updating Google Sheets with results")
    
    # Function to open the spreadsheet with retry
    def open_spreadsheet_with_retry(max_attempts=3):
        for attempt in range(1, max_attempts + 1):
            try:
                sheet = gc.open_by_key(spreadsheet_id)
                return sheet
            except Exception as e:
                logger.warning(f"Failed to open spreadsheet (attempt {attempt}/{max_attempts}): {e}")
                if attempt == max_attempts:
                    raise
                time.sleep(2 ** attempt)
    
    try:
        # Update batters worksheet
        try:
            sheet = open_spreadsheet_with_retry()
            batters_worksheet = sheet.worksheet('batters_by_month')
            batters_worksheet.clear()
            logger.info("Cleared batters worksheet")
        except Exception as e:
            sheet = open_spreadsheet_with_retry()
            logger.info(f"Creating new batters worksheet: {e}")
            batters_worksheet = sheet.add_worksheet(
                title='batters_by_month', 
                rows=len(batters_final)+1, 
                cols=len(batters_final.columns)
            )
        
        # Update batters data with retry
        batters_data = [batters_final.columns.values.tolist()] + batters_final.values.tolist()
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                batters_worksheet.update(batters_data)
                logger.info("Updated batters worksheet")
                break
            except Exception as e:
                logger.warning(f"Failed to update batters (attempt {attempt}/{max_attempts}): {e}")
                if attempt == max_attempts:
                    raise
                # Get a fresh connection to the worksheet before retrying
                sheet = open_spreadsheet_with_retry()
                batters_worksheet = sheet.worksheet('batters_by_month')
                time.sleep(2 ** attempt)
        
        # Update pitchers worksheet - reopening connection
        try:
            sheet = open_spreadsheet_with_retry()  # Fresh connection
            pitchers_worksheet = sheet.worksheet('pitchers_by_month')
            pitchers_worksheet.clear()
            logger.info("Cleared pitchers worksheet")
        except Exception as e:
            sheet = open_spreadsheet_with_retry()
            logger.info(f"Creating new pitchers worksheet: {e}")
            pitchers_worksheet = sheet.add_worksheet(
                title='pitchers_by_month', 
                rows=len(pitchers_final)+1, 
                cols=len(pitchers_final.columns)
            )
        
        # Update pitchers data with retry
        pitchers_data = [pitchers_final.columns.values.tolist()] + pitchers_final.values.tolist()
        for attempt in range(1, max_attempts + 1):
            try:
                pitchers_worksheet.update(pitchers_data)
                logger.info("Updated pitchers worksheet")
                break
            except Exception as e:
                logger.warning(f"Failed to update pitchers (attempt {attempt}/{max_attempts}): {e}")
                if attempt == max_attempts:
                    raise
                # Get a fresh connection to the worksheet before retrying
                sheet = open_spreadsheet_with_retry()
                pitchers_worksheet = sheet.worksheet('pitchers_by_month')
                time.sleep(2 ** attempt)
        
        # Update leaderboard worksheet - reopening connection
        try:
            sheet = open_spreadsheet_with_retry()  # Fresh connection
            leaderboard_worksheet = sheet.worksheet('Leaderboard')
            leaderboard_worksheet.clear()
            logger.info("Cleared leaderboard worksheet")
        except Exception as e:
            sheet = open_spreadsheet_with_retry()
            logger.info(f"Creating new leaderboard worksheet: {e}")
            leaderboard_worksheet = sheet.add_worksheet(
                title='Leaderboard', 
                rows=len(leaderboard)+1, 
                cols=len(leaderboard.columns)+3
            )
        
        # Update leaderboard data with retry
        leaderboard_data = [leaderboard.columns.values.tolist()] + leaderboard.values.tolist()
        for attempt in range(1, max_attempts + 1):
            try:
                leaderboard_worksheet.update(leaderboard_data)
                logger.info("Updated leaderboard worksheet")
                
                # Update timestamp
                eastern = pytz.timezone('America/New_York')
                eastern_now = datetime.now(eastern)
                today = eastern_now.date()
                yesterday = today - timedelta(days=1)
                leaderboard_worksheet.update_cell(1, 7, yesterday.isoformat())
                logger.info("Updated timestamp")
                break
            except Exception as e:
                logger.warning(f"Failed to update leaderboard (attempt {attempt}/{max_attempts}): {e}")
                if attempt == max_attempts:
                    raise
                # Get a fresh connection to the worksheet before retrying
                sheet = open_spreadsheet_with_retry()
                leaderboard_worksheet = sheet.worksheet('Leaderboard')
                time.sleep(2 ** attempt)
        
        logger.info("Google Sheets updated successfully")
        
    except Exception as e:
        logger.error(f"Error updating Google Sheets: {e}")
        raise
    
def get_game_ids(start_date, end_date):
    """
    Get MLB game IDs for a date range
    """
    logger.info(f"Getting game IDs from {start_date} to {end_date}")
    
    try:
        games = statsapi.schedule(start_date=start_date, end_date=end_date)
        game_ids = []
        
        for game in tqdm(games):
            if game['game_id'] not in game_ids:
                game_ids.append(game['game_id'])
        
        logger.info(f"Found {len(game_ids)} games")
        return game_ids
    except Exception as e:
        logger.error(f"Error getting game IDs: {e}")
        raise

def main():
    """
    Main function to run the fantasy baseball automation
    """
    try:
        # Get spreadsheet ID from environment
        spreadsheet_id = os.environ.get('SPREADSHEET_ID')
        if not spreadsheet_id:
            logger.error("SPREADSHEET_ID environment variable is not set")
            sys.exit(1)
            
        # Setup Google Sheets client
        gc = setup_google_sheets_client()
        
        # Get fantasy submissions
        submissions = get_fantasy_submissions(gc, spreadsheet_id)
        
        # Extract player names
        h_names, p_names = extract_player_names(submissions)
        
        # Look up player IDs
        player_info, h_id, p_id = lookup_player_ids(gc, spreadsheet_id, h_names, p_names)
        
        # Determine date range for games
        eastern = pytz.timezone('America/New_York')
        eastern_now = datetime.now(eastern)
        today = eastern_now.date()
        yesterday = today - timedelta(days=1)
        
        current_month = get_current_month()
        month_start = '2025-03-27' if today.month in (3, 4) else today.replace(day=1)
        
        # Get game IDs
        game_ids = get_game_ids(month_start, yesterday)
        
        # Get player stats
        batters = get_batter_stats(game_ids, h_id)
        pitchers = get_pitcher_stats(game_ids, p_id)
        
        # Process pitcher stats
        pitchers = process_pitcher_stats(pitchers)
        
        # Calculate fantasy points
        batters, pitchers = calculate_player_points(batters, pitchers)
        
        # Create player summaries
        batters_final, pitchers_final = create_player_summaries(batters, pitchers, player_info)
        
        # Create leaderboard
        leaderboard = create_leaderboard(submissions, batters_final, pitchers_final)
        
        # Update Google Sheets
        try:
            update_google_sheets(
                gc=gc,  # Use your existing gspread client
                spreadsheet_id=spreadsheet_id,
                batters_final=batters_final,
                pitchers_final=pitchers_final,
                leaderboard=leaderboard
            )
            logger.info("Fantasy baseball update completed successfully")
        except Exception as e:
            logger.error(f"Error in main function: {e}")
            
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()