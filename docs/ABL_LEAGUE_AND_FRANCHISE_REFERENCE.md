# Action Baseball League — League and Franchise Reference

Quick-reference source for ABL reporting and research, including franchise identity, league alignment, ballparks, organization staff, and affiliated clubs.

## Scope and date

This reference describes the ABL export in the repository with the main league (`league_id=200`) set to `current_date=1981-8-1`. Franchise identity and ballpark details come from the OOTP team and park records; staff assignments and farm affiliations are the assignments present in that same snapshot. Staff, affiliates, rules, and schedules can change over time, so treat those as **as of August 1, 1981**, not timeless facts. This is a source reference, not a claim that every person named held the role in another season.

Use ABL names exactly as listed here. The canonical display name is assembled from the trimmed `name` and `nickname` fields in `teams.csv`; a few source values contain stray outer whitespace, which is normalized here. Do not substitute a real-world or MLB franchise nickname.

## League at a glance

| Item | Repository record |
|---|---|
| League | Action Baseball League (ABL), OOTP league ID 200 |
| Earliest season represented in league-history exports | 1972 (also the league's `world_start_year`) |
| 1981 season opening date | April 6, 1981 |
| Major-league clubs | 24, IDs 1–24 |
| Structure | National Baseball Conference (NBC) and American Baseball Conference (ABC); each has Eastern, Central, and Western divisions, four clubs per division |
| Designated hitter | Enabled in both conferences |
| Scheduled games per club | 162 in the 1981 league rules |
| Typical series length | Three games in the 1981 league rules |
| 1981 All-Star Game | July 14, 1981 |
| 1981 trade deadline | July 31, 1981 |
| Active / secondary / expanded roster limits | 26 / 40 / 40 in the 1981 league export |

These are export values, not assertions about rules in every ABL season. NBC/ABC names, division names, dates, game count, roster limits, and DH setting are read from `leagues.csv`, `sub_leagues.csv`, and `divisions.csv`.

## Franchise and ballpark directory

Ballpark capacities are the values recorded in `parks.csv`; the city and state are joined from each team’s `city_id`.

| Team ID | Franchise | Code | Conference | Division | City, State | Home ballpark | Capacity |
|---:|---|---|---|---|---|---|---:|
| 1 | Miami Hurricanes | MIA | NBC | Eastern | Miami, Florida | Hurricanes Park | 25,800 |
| 2 | Phoenix Firebirds | PHO | NBC | Western | Phoenix, Arizona | Phoenix Grounds | 23,600 |
| 3 | Atlanta Kings | ATL | NBC | Eastern | Atlanta, Georgia | Kings Field | 23,300 |
| 4 | Dallas Rustlers | DAL | NBC | Central | Dallas, Texas | Rustlers Grounds | 23,400 |
| 5 | Detroit Dukes | DET | NBC | Central | Detroit, Michigan | Dukes Field | 28,100 |
| 6 | Charlotte Colonels | CHA | NBC | Eastern | Charlotte, North Carolina | Colonels Park | 26,000 |
| 7 | San Diego Seraphs | SD | NBC | Western | San Diego, California | Seraphs Field | 25,700 |
| 8 | Tampa Bay Storm | TB | NBC | Eastern | Tampa, Florida | Storm Stadium | 28,400 |
| 9 | Los Angeles Cobras | LA | NBC | Western | Los Angeles, California | Cobras Ballpark | 28,400 |
| 10 | San Francisco Warriors | SF | NBC | Western | San Francisco, California | San Francisco Grounds | 25,800 |
| 11 | Minneapolis Blizzard | MIN | NBC | Central | Minneapolis, Minnesota | Minneapolis Grounds | 23,300 |
| 12 | Chicago Fire | CHI | NBC | Central | Chicago, Illinois | Chicago Grounds | 18,500 |
| 13 | Denver Rocketeers | DEN | ABC | Western | Denver, Colorado | Rocketeers Field | 28,100 |
| 14 | Cincinnati Cougars | CIN | ABC | Central | Cincinnati, Ohio | Cougars Park | 21,000 |
| 15 | New York Aces | NY | ABC | Eastern | New York, New York | Aces Field | 25,700 |
| 16 | Nashville Blues | NAS | ABC | Central | Nashville, Tennessee | Blues Stadium | 21,200 |
| 17 | Las Vegas Gamblers | LV | ABC | Western | Las Vegas, Nevada | Gamblers Park | 20,900 |
| 18 | Portland Lumberjacks | POR | ABC | Western | Portland, Oregon | Portland Grounds | 20,900 |
| 19 | St. Louis Stallions | STL | ABC | Central | St. Louis, Missouri | Stallions Grounds | 26,000 |
| 20 | Boston Patriots | BOS | ABC | Eastern | Boston, Massachusetts | Patriots Fields | 26,000 |
| 21 | Pittsburgh Express | PIT | ABC | Eastern | Pittsburgh, Pennsylvania | Express Stadium | 25,700 |
| 22 | Philadelphia Fury | PHI | ABC | Eastern | Philadelphia, Pennsylvania | Fury Park | 21,200 |
| 23 | Houston Mavericks | HOU | ABC | Central | Houston, Texas | Houston Stadium | 21,000 |
| 24 | Seattle Comets | SEA | ABC | Western | Seattle, Washington | Seattle Stadium | 25,800 |

## Major-league organization directory

Staff names are the OOTP records assigned to each club in `team_roster_staff.csv`. The slot called `manager` is the field manager; `general_manager` is the GM. Names are followed by stable OOTP coach IDs in parentheses. The roster staff export also identifies a scout and doctor, included below as other organization staff.

| Franchise (ID) | Owner | General manager | Field manager | Pitching coach | Hitting coach | Bench coach | First-base coach | Third-base coach | Head scout | Team doctor |
|---|---|---|---|---|---|---|---|---|---|---|
| Miami Hurricanes (1) | Danny Leon (916) | Daniel Baker (34) | Seth Coe (258) | Ed Juarez (927) | David Powell (582) | Pontus Parnevik (500) | Arturo Guerra (228) | Marc Vergara (163) | Curt Tatro (25) | Josh Lester (24) |
| Phoenix Firebirds (2) | Mike Dutcher (858) | Bryan Somers (113) | Bill Benisch (37) | Fernando Chavez (90) | Elliot Bell (8) | Chris Doyle (353) | Tim McSherry (3) | Bill Hansen (809) | Malik Jones (88) | Roger Huck (87) |
| Atlanta Kings (3) | Tyler Carter (859) | Hector Naranjo (5) | T.J. Cameron (136) | Dave Pridgen (313) | Mario Molina (312) | Gene Nathan (19) | Leo Ramos (275) | Eddie Lyle (947) | Danny Castillo (7) | Matt Fletcher (6) |
| Dallas Rustlers (4) | Tommy Mauldin (860) | Jim Griffith (269) | Brian Parsley (146) | Dave Mendez (325) | Brad Sewell (242) | David Thorstenson (254) | Rich Ough (111) | Danny Quinones (284) | Chris O'Brien (52) | Luis Alvarez (168) |
| Detroit Dukes (5) | David Figueiredo (861) | Jesus Martinez (229) | Jose Perez (92) | Isaiah Boutin (775) | Eric Olson (62) | Aaron Hawkins (56) | Kent Weaver (39) | Alex Wagoner (58) | Eric Knittel (61) | Ken Klecker (60) |
| Charlotte Colonels (6) | Adam Butler (862) | Jose Ruiz (892) | Jason Wainwright (356) | Tony Brown (385) | Bill Carolina (929) | Jason Eatmon (234) | Jon Chandler (735) | Jace Doerr (13) | Alfonso Palma (16) | Jon Spangler (250) |
| San Diego Seraphs (7) | Danny Tocornal (863) | Matt Johnson (881) | Jesse Wade (143) | Rafael Alcantara (942) | Aaron Bankston (348) | Phil Leary (599) | Jeremy Rivera (954) | Chris Jervey (94) | Danny Amaya (97) | Chris Carter (96) |
| Tampa Bay Storm (8) | Jaylin Reed (864) | Angel Gonzalez (158) | Mike Khoury (941) | Ryan Thedford (448) | Erik Milstead (612) | Mike Tuss (488) | Chris Brooks (184) | Jon Blue (193) | Jason Davidson (918) | Phil Tavarez (33) |
| Los Angeles Cobras (9) | Tony Cruz (865) | Gary Jones (59) | Dave Walkling (653) | Danny Chambers (99) | Aaron Cummins (780) | Danny Hatch (173) | Jim Cicone (103) | Adam Trautman (286) | George Siemon (79) | Jonathon Tanner (78) |
| San Francisco Warriors (10) | Jon Teeple (866) | Luis Martinez (886) | Tyler Trowbridge (64) | Mike Richardson (108) | Kevin Garwood (330) | Phil Kuntz (255) | Luis Garcia (704) | Joe Sullivan (717) | Bobby Ordonez (106) | George Perry (230) |
| Minneapolis Blizzard (11) | Chad Purchase (867) | Ken Sears (887) | Shawn Thompson (915) | Jorge Toledo (920) | John Rodgers (459) | Eloy Batista (66) | Wayne Kielich (244) | Alan Poindexter (224) | Jesus Pimentel (901) | Mark Van Buren (69) |
| Chicago Fire (12) | Willie Roy (962) | Mike Hatton (41) | Matt Mead (46) | David Price (171) | Bob Moran (624) | Nick Babb (615) | Willie Gutierrez (850) | Kevin Durance (157) | Matt Buros (43) | Ignacio Martinez (42) |
| Denver Rocketeers (13) | Ken Ellinger (869) | Kevin Bradley (888) | Greg Asche (12) | Alex Garcia (847) | Edwin Perez (432) | Roger Tanton (706) | Eddy Rafael (820) | Fredrik Kofoed (730) | Ryan Hibbard (187) | Salvador Moreno (906) |
| Cincinnati Cougars (14) | Ji-hu Ju (914) | Phil Trammel (889) | Chris Allison (182) | Jon Baugher (117) | Jared Teitelbaum (152) | Scott Willett (1) | Joe Teague (256) | Dante Johnson (266) | Darryl Middaugh (151) | Mason Hocking (150) |
| New York Aces (15) | Alberto Dominguez (871) | Jeremy Burk (122) | Dennis Salvino (845) | Curt Brown (126) | Matt Brown (107) | Vince Schultz (119) | Mike Boyd (110) | Justin Anderson (121) | David Finch (124) | Al D'Amiano (123) |
| Nashville Blues (16) | Chris Smith (872) | Justin Gomez (32) | Francisco Hernandez (199) | Walt Elliott (198) | Chris Todd (188) | Steve Dubuque (332) | Adam Ross (165) | Keith Battista (928) | Mike Karlson (169) | Eric Ayers (220) |
| Las Vegas Gamblers (17) | Terry Vinson (873) | Roberto Choto (149) | Matt Petrie (218) | Larry Clark (135) | Dustin Cronkhite (759) | Justin Bartolomei (264) | Isaiah Ellington (192) | Aaron Green (21) | Andrew Mosley (271) | Jose Villegas (195) |
| Portland Lumberjacks (18) | Ben Criswell (874) | Ben Hayter (259) | Tim Staheli (638) | David Padilla (951) | Chris Hamilton (630) | Tony Mitchell (65) | Eric Baum (211) | Javy Navarro (100) | Ken Duckworth (205) | Ethan Wilson (946) |
| St. Louis Stallions (19) | J.D. Sommer (875) | David Ward (239) | Ryan Messmer (164) | Jason Hovest (784) | Giustiniano Misiano (342) | Nate Berlin (210) | Jose Morales (226) | Bobby Lee (921) | Nigel Timbers (178) | Jose Tirado (270) |
| Boston Patriots (20) | Alex Mendez (876) | Bobby Ramsey (203) | Bill Hurff (73) | Tyler Harrison (72) | Jun-young Kim (957) | Chris Sturdevant (48) | Austin Libin (597) | Josh Dalton (265) | Jon Thomas (115) | Ted Foxley (114) |
| Pittsburgh Express (21) | Jared Brown (877) | Nick Heller (212) | Mark Averill (510) | Kevin Piluso (331) | Jose Nava (222) | Dan Allen (710) | Roberto Sapon (138) | Mario Torres (923) | Justin McGovern (142) | Kevin Blanchard (141) |
| Philadelphia Fury (22) | Gabe Eckard (878) | Jon Ramos (891) | Danny Sanchez (470) | Jimmy Sewell (541) | Jim Coons (438) | Ryan Mendoza (83) | Bill Contreras (945) | John McCullough (832) | David Thayer (133) | Lazar Savicevic (944) |
| Houston Mavericks (23) | Xavier Ferguson (926) | Larry Bell (219) | Carlos Sanchez (238) | Rodolfo Luna (162) | Ray Gallo (549) | Sam Downs (842) | Paul Lacy (955) | Chris Jon (245) | Eric Metz (160) | Robbie Bryson (15) |
| Seattle Comets (24) | Tony Manning (880) | David Suarez (893) | William Ross (200) | Allen Snyder (263) | Jesse Welch (828) | Jared McDevitt (917) | Marco Cordero (285) | Ramon de los Santos (935) | Tony Matanillas (896) | Danny Grunza (260) |

## Farm systems / affiliated clubs

The affiliate relationship is the direction recorded in `team_affiliations.csv`: major-league `team_id` → `affiliated_team_id`. Each ABL club has one affiliate in each of the five minor-league classes in this snapshot. Classes are mapped from the affiliate's league in `leagues.csv`; club names are the ABL names, not real-world affiliations.

| Parent franchise (ID) | Triple A | Double A | Class A | Short Season A | Rookie League |
|---|---|---|---|---|---|
| Miami Hurricanes (1) | Arlington Patriots | Ackerman Classics | Fontana Velocity | Sanatoga Gold Rush | Baton Rouge Smokies |
| Phoenix Firebirds (2) | Spokane Comets | Cypress Sidewinders | Durham Brewers | Honolulu Ambassadors | St. Petersburg Elephants |
| Atlanta Kings (3) | Sunrise Manor Millers | Montgomery Eagles | Aurora Bruins | Amarillo Sky Sox | Terra Mar Tycoons |
| Dallas Rustlers (4) | Chula Vista Blackhawks | Gilbert Irish | Russellville Warcats | Enfield Tornado | Port Royal Predators |
| Detroit Dukes (5) | Jackson Stars | Little Rock Red Claws | Novato Padres | Middleton Hounds | Providence Players |
| Charlotte Colonels (6) | Glendale Blue Hens | Brooklyn Warriors | New Orleans Hooks | Easton Conquest | Boise Majestics |
| San Diego Seraphs (7) | Virginia Beach Aviators | Omaha Warlocks | Scottsdale Texans | Mire Mustangs | Knoxville Bayhawks |
| Tampa Bay Storm (8) | Sergeant Bluff Diamondbacks | Mesa Victory | Akron Honkers | Orlando Paladins | Hialeah War Dogs |
| Los Angeles Cobras (9) | Algoma Shamrocks | Fort Ashby Colonels | Birmingham Blue Falcons | Giddings Thunder Bats | Bonduel Mercenaries |
| San Francisco Warriors (10) | Grand Rapids Pipers | Paradise Badlanders | Stinnett Rail Cats | Rochester Thunder Dogs | Denmark Blue Falcons |
| Minneapolis Blizzard (11) | Salt Lake City Badlanders | Norfolk Stingrays | Laredo Checkers | Reno Swamp Dogs | Warren Bombers |
| Chicago Fire (12) | Santa Clarita Red Devils | Cherryvale Speed Devils | Madison Sundevils | Mountainair Manatees | Irving Volts |
| Denver Rocketeers (13) | South Bay Turtles | Lions Swamp Dogs | Weatherby Lake Alouettes | Elton Skyliners | Cantonment Rough Riders |
| Cincinnati Cougars (14) | Fort Lauderdale Diamond Dawgs | Modesto Americans | Kansas City Spirit | Eufaula Valiants | Kentland Nuggets |
| New York Aces (15) | Henderson Sailors | Stockton Mountaineers | Oxnard Legends | Fort Wayne Lynx | Richmond Torpedoes |
| Nashville Blues (16) | Wilkinson Heights Tyrants | Seaside Broncos | St. Paul Devil Rays | Jersey City White Knights | St. Louis Conquistadors |
| Las Vegas Gamblers (17) | Des Moines Caribou | Chandler Seagulls | Lexington Black Dogs | Arabi Superstars | North Hempstead Twins |
| Portland Lumberjacks (18) | Santa Ana Lone Stars | San Felipe Pueblo River Otters | Lincoln Village Federals | Oak Point Stampede | Johnson Bayou Tides |
| St. Louis Stallions (19) | Worcester Bucs | Louisville Lancers | Winston-Salem Isotopes | Lydia 76ers | Bakersfield Bolts |
| Boston Patriots (20) | Buffalo Texans | Waynesville Celtics | Gardner Thrashers | Knoxville Blue Knights | Fremont Marksmen |
| Pittsburgh Express (21) | Shreveport Oceanics | Lubbock Lords | Colorado Springs Kernels | Ladysmith Road Warriors | Waterford Whalers |
| Philadelphia Fury (22) | Glendale Owls | Fairbanks Ranch Emperors | Florence Tiger Sharks | Kemper Impact | Onarga Veterans |
| Houston Mavericks (23) | Arlington Bruins | Lincoln Honu | Anaheim Klondikes | Haynesville Attack | Poolville Broncos |
| Seattle Comets (24) | Mobile Voyagers | Newark Kernels | Irvine Dingos | Kemmerer Diamond Miners | Rodessa Beacons |

## Source map and use notes

- `csv/ootp_csv/leagues.csv` — league names and IDs, current date, 1981 schedule/rules, roster limits, all-star date, trade deadline, league level.
- `csv/ootp_csv/sub_leagues.csv` and `csv/ootp_csv/divisions.csv` — conference and division names.
- `csv/ootp_csv/teams.csv` — team IDs, canonical name components, codes, city/park IDs, conference/division IDs, and affiliate team records.
- `csv/ootp_csv/cities.csv` and `csv/ootp_csv/states.csv` — city and state names for team locations.
- `csv/ootp_csv/parks.csv` — park names, capacity, and park metadata.
- `csv/ootp_csv/team_roster_staff.csv` — role-to-staff-ID assignments by major-league team.
- `csv/ootp_csv/coaches.csv` — names and stable OOTP coach IDs for owner, GM, manager, and staff records.
- `csv/ootp_csv/team_affiliations.csv` — major-club to affiliate team-ID links.
- `csv/ootp_csv/teams.csv` plus `leagues.csv` — affiliate club names and their league/class.

For article claims about a particular date, use contemporaneous records for standings, transactions, availability, or performance. This reference answers **who/what/where** in the August 1, 1981 export; it does not establish private owner instructions, staff motives, player availability on another date, or that affiliations and staff remained unchanged later.
