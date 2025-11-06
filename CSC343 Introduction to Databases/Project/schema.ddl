drop schema if exists Champion cascade;
create schema Champion;
set search_path to Champion;

-- Create domain for 10 seasons
create domain Season as varchar(3)
	not null
	check (value in ('S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10'));

-- Create domain for 5 different positions in the game
create domain Posi as varchar(3)
	not null
	check (value in ('top', 'jgl', 'mid', 'bot', 'sup'));

-- Create domain for type of matches
create domain Type as varchar(9)
	not null
	check (value in ('Final', 'Semifinal'));

-- Create domain for 4 different regions
create domain Region as varchar(13)
	not null
	check (value in ('Americas', 'Europe', 'Asia', 'International'));

-- Create table for team
create table Team(
    -- abbreviation of full team name as team ID
	tID varchar(6) primary key,
	-- full name of the team
	teamName varchar(30) not null,
	-- the year where this team is established
	establishYear integer not null,
	-- whether this team still exists
	exist boolean not null);

create table League(
    -- the name of team
	tID varchar(6) primary key,
    -- abbreviation of full league name as league ID
	lID varchar(5),
	-- full name of the league
	leagueName varchar(50) not null,
	-- the corresponding region this league is in
	region Region,
	foreign key (tID) references Team (tID));
	
create table Player(
    -- an integer as player ID
	pID integer primary key,
	-- name of this player
	name varchar(40) not null,
	-- birth year of this player, 0 if no record
	birth integer not null);
	
create table WorldChampion(
    -- season of this world championship
	season Season,
	-- year of this world championship
	year integer check (year >= 2011 and year <= 2020),
	-- team ID of the champion
	champion varchar(6) not null,
	-- team ID of the second place
	secondPlace varchar(6) not null,
	primary key (season),
	foreign key (champion) references Team (tID),
	foreign key (secondPlace) references Team (tID));

create table Register(
    -- player ID
	pID integer not null,
	-- team ID which this player is joined
	tID varchar(6) not null,
	-- the season which this player is in this team
	season Season,
	-- the position which this player is at
	position Posi,
	primary key (pID, tID, season, position),
	foreign key (pID) references Player (pID),
	foreign key (tID) references Team (tID),
	foreign key (season) references WorldChampion (season));

create table Matches(
    -- an integer for this match as match ID
	mID integer primary key,
	-- season which this match happened
	season Season,
	-- two team ID arranged in alphabetical order
	team1 varchar(6),
	team2 varchar(6),
	-- two scores that these teams earned
	score1 integer check (score1 >= 0 and score1 <= 3),
	score2 integer check (score2 >= 0 and score2 <= 3 and score1 + score2 <= 5),
	-- type of this match
	type Type,
	foreign key (team1) references Team (tID),
	foreign key (team2) references Team (tID),
	foreign key (season) references WorldChampion (season))
	-- every match contains 5 games so the team wins 3 games first wins this match;
