SET search_path TO Champion;

-- Q1. Who are the youngest players that have won a world champion?
-- Return players' name, their age at that time and the League they
-- were from.
-- Drop tables here
DROP TABLE IF EXISTS youngest CASCADE;
DROP TABLE IF EXISTS oldest CASCADE;

-- Create tables
CREATE TABLE youngest (
    name varchar(40) NOT NULL,
    age INT NOT NULL,
    league varchar(5) NOT NULL
);

CREATE TABLE oldest (
    name varchar(40) NOT NULL,
    age INT NOT NULL,
    league varchar(5) NOT NULL
);


-- Drop views here
DROP VIEW IF EXISTS ChampionPlayers CASCADE;
DROP VIEW IF EXISTS PlayerAge CASCADE;

-- Define views for intermediate steps here
-- Find out all the players that win champion
CREATE VIEW ChampionPlayers AS
SELECT wc.year, r.pID, r.tID
FROM WorldChampion wc JOIN Register r
    ON wc.season = r.season
WHERE wc.champion = r.tID;

-- Calculate those players age and their league
CREATE VIEW PlayerAge AS
SELECT p.name, (cp.year - p.birth) as age, l.lID
FROM ChampionPlayers cp JOIN Player p
        ON cp.pID = p.pID
    JOIN League l
        ON cp.tID = l.tID;

-- Insert tables here
INSERT INTO youngest
SELECT *
FROM PlayerAge pa1
WHERE pa1.age = (
    SELECT min(pa2.age)
    FROM PlayerAge pa2);

INSERT INTO oldest
SELECT *
FROM PlayerAge pa1
WHERE pa1.age = (
    SELECT max(pa2.age)
    FROM PlayerAge pa2)
    AND pa1.age <= 30;



-- Q2. Which two teams have the most fate with each other?
-- In other words, which two teams meet each other mostly
-- in World Championship matches and how many times do they
-- meet?
-- Drop tables here
DROP TABLE IF EXISTS fate CASCADE;

-- Create tables here
CREATE TABLE fate (
    teamName1 varchar(6) NOT NULL,
    teamName2 varchar(6) NOT NULL,
    matches INT NOT NULL
);

-- Drop views here
DROP VIEW IF EXISTS SimilarMatch CASCADE;

-- Define views for intermediate steps here
-- Find all the matches that these two teams met
-- multiple times
CREATE VIEW SimilarMatch AS
SELECT m1.team1, m1.team2, count(m1.team1 || m1.team2)
FROM Matches m1, Matches m2
WHERE m1.mID <> m2.mID
    AND ((m1.team1 = m2.team1
            AND m1.team2 = m2.team2)
        OR (m1.team1 = m2.team2
            AND m1.team2 = m2.team1))
group by m1.team1 || m1.team2, m1.team1, m1.team2;

-- Insert tables here
INSERT INTO fate
SELECT *
FROM SimilarMatch;



-- Q3 What is the name of the team that faced the most “tough games”
-- in World Championships? “Tough game” is a match in which two teams
-- have competed for the entire five rounds. Use each team's full
-- name.
-- Drop tables here
DROP TABLE IF EXISTS ToughGame CASCADE;

-- Create tables here
CREATE TABLE toughgame (
    teamName varchar(30) NOT NULL,
    toughGame INT NOT NULL
);

-- Drop views here
DROP VIEW IF EXISTS ToughMatches CASCADE;
DROP VIEW IF EXISTS ToughTeam CASCADE;

-- Define views for intermediate steps here
-- Find all the tough matches
CREATE VIEW ToughMatches AS
SELECT *
FROM Matches
WHERE score1 + score2 = 5;

-- Find all the teams in those matches
CREATE VIEW ToughTeam AS
SELECT teamName
FROM ToughMatches tm, Team t
where t.tID=tm.team1 or t.tID=tm.team2;

-- Insert tables here
INSERT INTO toughgame
SELECT tt.teamName, count(tt.teamName) as num
FROM ToughTeam tt
GROUP BY tt.teamName
ORDER BY num DESC;


