# Data audit: Balkan flight-delay dataset

- Records: **300,738**
- Destination airports: **10**
- Unique origins: **790**
- Routes: **3186**
- Period (UTC): **2015-03-01T05:14:43+00:00 – 2023-06-30T22:07:55+00:00**
- Delayed at least 15 minutes: **14.458%**
- Missing core rows: **0**
- Rows participating in an exact duplicate: **12**
- Destination/file mismatches: **0**
- Extreme-delay flags: **12**

Extreme-delay rows are flagged for review, not removed. The initial time features use
UTC because a verified local-timezone mapping has not yet been added for every airport.
No ML model is trained by this audit.
