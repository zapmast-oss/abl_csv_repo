# ABL Team Finances

| Team | Budget | Payroll | Cash | Revenue | Balance | Tier |
| --- | --- | --- | --- | --- | --- | --- |
| ATL | $13.0m | $8.6m | $0 | $8.5m | $4,777,472 | N/A |
| LV | $6.5m | $5.5m | $0 | $5.8m | $1,001,756 | N/A |
| HOU | $9.6m | $8.4m | $-370,345 | $8.9m | $4,579,491 | N/A |
| BOS | $10.0m | $8.4m | $-81,691 | $10.7m | $4,412,551 | N/A |
| SEA | $12.3m | $7.2m | $0 | $9.7m | $6,316,648 | N/A |
| CIN | $11.9m | $11.2m | $-40,667 | $11.3m | $4,135,699 | N/A |
| SF | $9.5m | $6.2m | $-397,214 | $6.8m | $233,070 | N/A |
| DET | $11.4m | $9.6m | $-550,297 | $14.0m | $9,257,486 | N/A |
| LA | $6.3m | $5.7m | $0 | $5.4m | $88,267 | N/A |
| NY | $11.3m | $7.5m | $-302,852 | $9.3m | $4,414,493 | N/A |
| PIT | $7.5m | $7.6m | $-6,099 | $5.5m | $921,448 | N/A |
| NAS | $13.4m | $8.2m | $0 | $10.4m | $2,557,426 | N/A |
| CHI | $10.7m | $8.4m | $-920,321 | $7.5m | $3,026,890 | N/A |
| CHA | $11.2m | $9.0m | $-115,691 | $9.2m | $5,138,291 | N/A |
| PHO | $15.6m | $13.2m | $-1,244,790 | $11.2m | $3,171,790 | N/A |
| TB | $13.0m | $10.9m | $-138,506 | $9.1m | $2,614,967 | N/A |
| PHI | $11.6m | $8.6m | $-307,605 | $7.9m | $2,372,369 | N/A |
| DAL | $7.8m | $5.4m | $-599,790 | $7.6m | $3,225,946 | N/A |
| SD | $6.6m | $4.5m | $-79,037 | $4.4m | $191,053 | N/A |
| POR | $6.4m | $6.3m | $-20,531 | $6.2m | $2,142,398 | N/A |
| DEN | $8.2m | $7.6m | $-116,395 | $8.9m | $1,313,638 | N/A |
| MIA | $9.5m | $7.7m | $-27,654 | $10.2m | $5,426,671 | N/A |
| MIN | $7.0m | $4.8m | $-72,729 | $6.4m | $2,437,585 | N/A |
| STL | $9.0m | $6.3m | $0 | $7.9m | $2,263,676 | N/A |

## Data Sources
- C:\sbv_repo\abl_csv_repo\csv\out\star_schema\fact_team_financials.csv
- C:\sbv_repo\abl_csv_repo\csv\ootp_csv\team_last_financials.csv

## Notes
- fact_team_financials columns: {'budget': 'Bgt', 'payroll': 'Pay', 'cash': None, 'revenue': 'Revenue', 'balance': None}
- team_last_financials.csv columns: {'budget': 'budget', 'payroll': None, 'cash': 'cash', 'revenue': 'total_revenue', 'balance': 'financial_balance'}
- team_history_financials load error: 'year'
- Budget Board omitted (insufficient numeric payroll/budget).
- Balance is sourced from financial_balance (OOTP export field name).
