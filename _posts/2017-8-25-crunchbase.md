---
layout: post
title: Database Schema of Crunchbase 2013 Snapshot.
---

[Crunchbase](https://www.crunchbase.com) has provided free access to 2013 snapshot of the database in their [website](https://data.crunchbase.com/v3/docs/2013-snapshot).

The schema of the database has been defined as:

# cb_objects

`cb_objects` is the most basic element in the database, which can be used to represent `Person`, `Company` and `Product`. This information can be found in field `entity_type`. `id` field is used as external index by other tables.
```
CREATE TABLE "cb_objects" (
  "id" varchar(64) NOT NULL,
  "entity_type" varchar(16) NOT NULL,
  "entity_id" bigint(20) NOT NULL,
  "parent_id" varchar(64) default NULL,
  "name" varchar(255) NOT NULL,
  "normalized_name" varchar(255) NOT NULL,
  "permalink" varchar(255) NOT NULL,
  "category_code" varchar(32) default NULL,
  "status" varchar(32) default 'operating',
  "founded_at" date default NULL,
  "closed_at" date default NULL,
  "domain" varchar(64) default NULL,
  "homepage_url" varchar(64) default NULL,
  "twitter_username" varchar(64) default NULL,
  "logo_url" varchar(255) default NULL,
  "logo_width" int(11) default NULL,
  "logo_height" int(11) default NULL,
  "short_description" varchar(255) default NULL,
  "description" varchar(255) default NULL,
  "overview" text,
  "tag_list" varchar(255) default NULL,
  "country_code" varchar(64) default NULL,
  "state_code" varchar(64) default NULL,
  "city" varchar(64) default NULL,
  "region" varchar(255) default NULL,
  "first_investment_at" date default NULL,
  "last_investment_at" date default NULL,
  "investment_rounds" int(11) default NULL,
  "invested_companies" int(11) default NULL,
  "first_funding_at" date default NULL,
  "last_funding_at" date default NULL,
  "funding_rounds" int(11) default NULL,
  "funding_total_usd" decimal(15,0) default NULL,
  "first_milestone_at" date default NULL,
  "last_milestone_at" date default NULL,
  "milestones" int(11) default NULL,
  "relationships" int(11) default NULL,
  "created_by" varchar(64) default NULL,
  "created_at" datetime default NULL,
  "updated_at" datetime default NULL,
  PRIMARY KEY  ("id"),
  UNIQUE KEY "entity" ("entity_type","entity_id"),
  KEY "permalink" ("permalink"),
  KEY "name" ("name"),
  KEY "normalized_name" ("normalized_name"),
  KEY "domain" ("domain")
);
```

# cb_relationships

`cb_relationships` represents relationship (employments) between a `Person` (referred as `person_object_id`) and `Company` (referred as `relationship_object_id`).

```
CREATE TABLE "cb_relationships" (
  "id" bigint(20) NOT NULL,
  "relationship_id" bigint(20) NOT NULL,
  "person_object_id" varchar(64) NOT NULL,
  "relationship_object_id" varchar(64) NOT NULL,
  "start_at" date default NULL,
  "end_at" date default NULL,
  "is_past" tinyint(4) default NULL,
  "sequence" int(11) default '0',
  "title" varchar(255) default NULL,
  "created_at" datetime default NULL,
  "updated_at" datetime default NULL,
  PRIMARY KEY  ("id"),
  KEY "person_object" ("person_object_id"),
  KEY "relationship_object" ("relationship_object_id")
);
```

# cb_people
`cb_people` represents the extra information of `Person` besides `cb_objects`, which is referred as `object_id`.

```
CREATE TABLE "cb_people" (
  "id" bigint(20) NOT NULL,
  "object_id" varchar(64) NOT NULL,
  "first_name" varchar(128) NOT NULL,
  "last_name" varchar(128) NOT NULL,
  "birthplace" varchar(128) default NULL,
  "affiliation_name" varchar(128) default NULL,
  PRIMARY KEY  ("id"),
  UNIQUE KEY "object" ("object_id")
);
```

# cb_offices
`cb_offices` represents the address information of objects, which is referred as `object_id`.

```
CREATE TABLE "cb_offices" (
  "id" bigint(20) NOT NULL,
  "object_id" varchar(64) NOT NULL,
  "office_id" bigint(20) NOT NULL,
  "description" varchar(255) default NULL,
  "region" varchar(255) default NULL,
  "address1" varchar(255) default NULL,
  "address2" varchar(255) default NULL,
  "city" varchar(255) default NULL,
  "zip_code" varchar(255) default NULL,
  "state_code" varchar(3) default NULL,
  "country_code" varchar(3) default NULL,
  "latitude" decimal(15,10) default NULL,
  "longitude" decimal(15,10) default NULL,
  "created_at" datetime default NULL,
  "updated_at" datetime default NULL,
  PRIMARY KEY  ("id"),
  KEY "object_id" ("object_id")
);
```
# cb_milestones

`cb_milestones` represents an important information for `Company`, which is referred as `object_id`.

```
CREATE TABLE "cb_milestones" (
  "id" bigint(20) NOT NULL,
  "object_id" varchar(64) NOT NULL,
  "milestone_at" date default NULL,
  "milestone_code" varchar(32) default NULL,
  "description" varchar(255) default NULL,
  "source_url" varchar(255) default NULL,
  "source_description" varchar(255) default NULL,
  "created_at" datetime default NULL,
  "updated_at" datetime default NULL,
  PRIMARY KEY  ("id"),
  KEY "object" ("object_id")
);
```

# cb_ipos
`cb_ipos` represents the IPO information of a `Company`, which is represented as `object_id`.

```
CREATE TABLE "cb_ipos" (
  "id" bigint(20) NOT NULL,
  "ipo_id" bigint(20) NOT NULL,
  "object_id" varchar(64) NOT NULL,
  "valuation_amount" decimal(15,0) default NULL,
  "valuation_currency_code" varchar(16) default NULL,
  "raised_amount" decimal(15,0) default NULL,
  "raised_currency_code" varchar(16) default NULL,
  "public_at" date default NULL,
  "stock_symbol" varchar(32) default NULL,
  "source_url" varchar(255) default NULL,
  "source_description" varchar(255) default NULL,
  "created_at" datetime default NULL,
  "updated_at" datetime default NULL,
  PRIMARY KEY  ("id"),
  KEY "object" ("object_id")
);
```

# cb_investments
`cb_investments` represents the investment information of `Person` or `Company` (referred as `investor_object_id`) to `Company` (referred as `funded_object_id`). The detailed financial information is included in `cb_funding_rounds` and referred as `funding_round_id`.

```
CREATE TABLE "cb_investments" (
  "id" bigint(20) NOT NULL,
  "funding_round_id" bigint(20) NOT NULL,
  "funded_object_id" varchar(64) NOT NULL,
  "investor_object_id" varchar(64) NOT NULL,
  "created_at" datetime default NULL,
  "updated_at" datetime default NULL,
  PRIMARY KEY  ("id"),
  KEY "funding_round" ("funding_round_id"),
  KEY "investor_object" ("investor_object_id"),
  KEY "funded_object" ("funded_object_id")
);
```

# cb_funding_rounds
`cb_funding_rounds` represents the detailed financial information in the investments.

```
CREATE TABLE "cb_funding_rounds" (
  "id" bigint(20) NOT NULL,
  "funding_round_id" bigint(20) NOT NULL,
  "object_id" varchar(64) NOT NULL,
  "funded_at" date default NULL,
  "funding_round_type" varchar(32) default NULL,
  "funding_round_code" varchar(32) default NULL,
  "raised_amount_usd" decimal(15,0) default NULL,
  "raised_amount" decimal(15,0) default NULL,
  "raised_currency_code" varchar(3) default NULL,
  "pre_money_valuation_usd" decimal(15,0) default NULL,
  "pre_money_valuation" decimal(15,0) default NULL,
  "pre_money_currency_code" varchar(3) default NULL,
  "post_money_valuation_usd" decimal(15,0) default NULL,
  "post_money_valuation" decimal(15,0) default NULL,
  "post_money_currency_code" varchar(3) default NULL,
  "participants" int(11) default NULL,
  "is_first_round" int(11) default '0',
  "is_last_round" int(11) default '0',
  "source_url" varchar(255) default NULL,
  "source_description" varchar(255) default NULL,
  "created_by" varchar(64) default NULL,
  "created_at" datetime default NULL,
  "updated_at" datetime default NULL,
  PRIMARY KEY  ("id"),
  KEY "object" ("object_id"),
  KEY "funding_round_id" ("funding_round_id")
);
```

# cb_funds
`cb_funds` contains the financial information of a `Company`, which is referred as `object_id`.

```
CREATE TABLE "cb_funds" (
  "id" bigint(20) NOT NULL,
  "fund_id" bigint(20) NOT NULL,
  "object_id" varchar(64) NOT NULL,
  "name" varchar(255) NOT NULL,
  "funded_at" date default NULL,
  "raised_amount" decimal(15,0) default NULL,
  "raised_currency_code" varchar(3) default NULL,
  "source_url" varchar(255) default NULL,
  "source_description" varchar(255) default NULL,
  "created_at" datetime default NULL,
  "updated_at" datetime default NULL,
  PRIMARY KEY  ("id"),
  KEY "object" ("object_id"),
  KEY "fund_id" ("fund_id")
);
```

# cb_degrees
`cb_degrees` contains the education information of `Person`, which is referred as `object_id`.

```
CREATE TABLE "cb_degrees" (
  "id" bigint(20) NOT NULL,
  "object_id" varchar(64) NOT NULL,
  "degree_type" varchar(32) NOT NULL,
  "subject" varchar(255) default NULL,
  "institution" varchar(64) default NULL,
  "graduated_at" date default NULL,
  "created_at" datetime default NULL,
  "updated_at" datetime default NULL,
  PRIMARY KEY  ("id"),
  KEY "person" ("object_id")
);
```

# cb_acquisitions
`cb_acquisitions` contains the acquisition information of a `Company` (referred as `acquired_object_id`) by a `Company` or `Person` (referred as `acquiring_object_id`).

```
CREATE TABLE "cb_acquisitions" (
  "id" bigint(20) NOT NULL,
  "acquisition_id" bigint(20) NOT NULL,
  "acquiring_object_id" varchar(64) NOT NULL,
  "acquired_object_id" varchar(64) NOT NULL,
  "term_code" varchar(16) default NULL,
  "price_amount" decimal(15,0) default NULL,
  "price_currency_code" varchar(16) default NULL,
  "acquired_at" date default NULL,
  "source_url" varchar(255) default NULL,
  "source_description" varchar(255) default NULL,
  "created_at" datetime default NULL,
  "updated_at" datetime default NULL,
  PRIMARY KEY  ("id"),
  KEY "acquiring_object_id" ("acquiring_object_id"),
  KEY "acquired_object_id" ("acquired_object_id"),
  KEY "acquisition_id" ("acquisition_id")
);
```
