use APD

--================================================
--[dbo].[tbl_Budgets]
--================================================
-- 1. Primary storage as columnstore (best for aggregations)
CREATE CLUSTERED COLUMNSTORE INDEX CI_Budgets
ON tbl_Budgets

-- 2. Add ONE nonclustered rowstore index for CaseNo lookups
CREATE NONCLUSTERED INDEX IX_Budgets_CaseNo
ON tbl_Budgets (CaseNo)
INCLUDE (FiscalYear, BudgetType, BudgetStatus, Programs, BudgetAmount, 
		AnnualizedAmount, AmountEncumbered, AmountUnauthorized, PrioriBudgetAmount, DateTimeStamp)

-- 3. Create unique non-clustered index for business key
CREATE UNIQUE NONCLUSTERED INDEX IX_Budgets_BusinessKey
ON tbl_Budgets (CaseNo, FiscalYear, BudgetType, BudgetStatus, Programs, BudgetAmount, 
		AnnualizedAmount, AmountEncumbered, AmountUnauthorized, PrioriBudgetAmount, DateTimeStamp)

/*
-- By trial-error, the following query returned 219,457 records, 
--which is ther size of the table. Therefore, this is the PK

SELECT COUNT(*) AS DistinctCombinations
FROM (
    SELECT DISTINCT CaseNo,  BudgetStatus, FiscalYear
    FROM tbl_Budgets
) AS UniqueRows
*/

--================================================
-- [dbo].[tbl_Claims_MMIS]
--================================================
-- 1. Primary storage as columnstore (best for aggregations)
CREATE CLUSTERED COLUMNSTORE INDEX CI_Claims_MMIS
ON tbl_Claims_MMIS

-- 2. Add ONE nonclustered rowstore index for CaseNo lookups
CREATE NONCLUSTERED INDEX IX_Claims_CaseNo
ON tbl_Claims_MMIS (CaseNo)
INCLUDE (ProcCode, ServiceDate, ICN, AdjustICN, ClaimType, ClaimSubType, LineNmbr)

-- 3. Create unique non-clustered index for business key
CREATE UNIQUE NONCLUSTERED INDEX IX_Claims_MMIS_BusinessKey
ON tbl_Claims_MMIS (CaseNo, ProcCode, ServiceDate, ICN, 
                    AdjustICN, ClaimType, ClaimSubType, LineNmbr)

/*
-- By trial-error, the following query returned 37,750,736 records, 
--which is ther size of the table. Therefore, this is the PK

SELECT COUNT(*) AS DistinctCombinations
FROM (
    SELECT DISTINCT CaseNo, ProcCode, ServiceDate, ICN, AdjustICN, ClaimType, ClaimSubType, LineNmbr
    FROM tbl_Claims_MMIS
) AS UniqueRows
*/

--================================================
-- [dbo].[tbl_ConsumerContacts]
--================================================
-- 1. Primary 
ALTER TABLE tbl_ConsumerContacts
ADD CONSTRAINT PK_ConsumerContacts PRIMARY KEY CLUSTERED (RECID)

-- 2. Add ONE nonclustered rowstore index for CaseNo lookups
CREATE NONCLUSTERED INDEX IX_ConsumerContacts_CaseNo
ON tbl_ConsumerContacts (CaseNo)
INCLUDE (CONTACTID, FIRSTNAME, LASTNAME, GENDER,   
RELATIONSHIP, Multirelationship, Active, DateTimeStamp, UserStamp, RECID)

-- 3. Create unique non-clustered index for business key
CREATE UNIQUE NONCLUSTERED INDEX IX_ConsumerContacts_BusinessKey
ON tbl_ConsumerContacts (CaseNo, CONTACTID, FIRSTNAME, LASTNAME, GENDER,   
RELATIONSHIP, Active, DateTimeStamp, UserStamp, RECID)

/*
-- By trial-error, the following query returned 433,650 records, 
--which is ther size of the table. Therefore, this is the PK

SELECT COUNT(*) AS DistinctCombinations
FROM (
    SELECT DISTINCT CONTACTID, FIRSTNAME, LASTNAME, GENDER, CaseNo,  RELATIONSHIP, Multirelationship, Active, DateTimeStamp, UserStamp, RECID
    FROM tbl_ConsumerContacts
) AS UniqueRows
*/


--================================================
-- [dbo].[tbl_Consumers]
--================================================
-- 1. Primary 
ALTER TABLE tbl_Consumers
ADD CONSTRAINT PK_Consumers PRIMARY KEY CLUSTERED (CaseNo)

-- 2. Add ONE nonclustered rowstore index for CaseNo lookups
CREATE NONCLUSTERED INDEX IX_Consumer_CaseNo
ON tbl_Consumers (CaseNo)
INCLUDE (Id,DOB,GENDER,RACE,PLANGUAGE,SLANGUAGE,TITLE,ETHNICITYLOOKUP,County,
District,Region,DOD,CauseOfDeath,DODAction,DODFileNumber,RESIDENCETYPE,
MedicaidId,ABCPIN,ReferralSource,CBCFlag,ReferredToVR,ANNUALINCOME,Competency,
[Status],DevelopmentalDisability,FUNDCODE,DISPOSITION,DISPOSITIONDATE,OPENDATE,
OPENREASON,CLOSEDATE,CLOSEREASON,ApplicationReceivedDate,ApplicationReceivedViaOAS,
ApplicantRequestingCWE,RequiresSOPTReview,DateAssignedToSOPT,SOPTName,
DateSOPTCompletedReview,OPENID,PRIMARYWORKER,PRIMARYWORKERID,SECONDWORKER,
SECONDWORKERID,PrimaryDiagnosis,SecondaryDiagnosis,OtherDiagnosis,
MentalHealthDiag1,MentalHealthDiag2,MentalHealthDiag3,MentalHealthDiag4,
MentalHealthDiag_5_6,REVIEW,REVIEWDATE,SSNMonthlyBenefitAmt,[3rdPartyHealthInsurance],
CompetitivelyEmployed,HireDate,AvgMonthlyEarnings,WantsEmployment,HourlyWage,MinimumWage,CONTACTID,DateTimeStamp,UserStamp)


--================================================
--[dbo].[tbl_Diagnosis]
--================================================
-- 1. Primary 
ALTER TABLE tbl_Diagnosis
ADD CONSTRAINT PK_Diagnosis PRIMARY KEY CLUSTERED (DiagnosisID)

-- 2. Add ONE nonclustered rowstore index for CaseNo lookups
CREATE NONCLUSTERED INDEX IX_Diagnosis_CaseNo
ON tbl_Diagnosis (CaseNo)
INCLUDE (STATUS, FUNDCODE, [PrimaryDiagnosis], [SecondaryDiagnosis], [TertiaryDiagnosis], [QuaternaryDiagnosis], REVIEW)

-- 3. Create unique non-clustered index for business key
CREATE UNIQUE NONCLUSTERED INDEX IX_Diagnosis_BusinessKey
ON tbl_Diagnosis (CaseNo, DiagnosisID, STATUS, FUNDCODE, [PrimaryDiagnosis], [SecondaryDiagnosis], [TertiaryDiagnosis], [QuaternaryDiagnosis], REVIEW)

/*
-- By trial-error, the following query returned 74,826 records, 
-- which is ther size of the table. Therefore, this is the PK

SELECT COUNT(*) AS DistinctCombinations
FROM (
    SELECT DISTINCT DiagnosisID
    FROM tbl_Diagnosis
) AS UniqueRows
*/

--================================================
--[dbo].[dbo].[tbl_EZBudget]
--================================================
-- 1. Primary 
ALTER TABLE tbl_EZBudget
ADD CONSTRAINT PK_EZBudget PRIMARY KEY CLUSTERED (EZBudgetAssessId)

-- 2. Add ONE nonclustered rowstore index for CaseNo lookups
CREATE NONCLUSTERED INDEX IX_EZBudget_CaseNo
ON tbl_EZBudget (CaseNo)
INCLUDE (REVIEW,Worker,ReviewDate,STATUS,Division,ApprovedBy,ApprovedDate,Region,UpdateSituation,LivingSetting,CurrentAge,PropFactor,QSIBehavioralScore
,QSIFunctionalScore,Q14,Q15,Q16,Q17,Q18,Q19,Q20,Q21,Q22,Q23,Q24,Q25,Q26,Q27,Q28,Q29,Q30,Q33,Q34,Q36,Q43,Q44,DATETIMESTAMP,UserStamp,EZBudgetAssessId,AlgorithmAmt)

-- 3. Create unique non-clustered index for business key
CREATE UNIQUE NONCLUSTERED INDEX IX_EZBudget_BusinessKey
ON tbl_EZBudget (CASENO,REVIEW,Worker,ReviewDate,STATUS,Division,ApprovedBy,ApprovedDate,Region,UpdateSituation,LivingSetting,CurrentAge,PropFactor,QSIBehavioralScore
,QSIFunctionalScore,DATETIMESTAMP,UserStamp,EZBudgetAssessId,AlgorithmAmt)

/*
-- By trial-error, the following query returned 43,213 records, 
-- which is ther size of the table. Therefore, this is the PK

SELECT COUNT(*) AS DistinctCombinations
FROM (
    SELECT DISTINCT CaseNo, REVIEW, Worker, ReviewDate, ApprovedBy, ApprovedDate, DATETIMESTAMP, UserStamp
    FROM tbl_EZBudget
) AS UniqueRows

SELeCT DISTINCT AlgorithmAmt FROM tbl_EZBudget
*/


--================================================
-- [dbo].[tbl_PlannedServices]
--================================================
-- 1. Primary 
ALTER TABLE tbl_PlannedServices
ADD CONSTRAINT PK_PlannedServices PRIMARY KEY CLUSTERED (PlannedServiceId)

-- 2. Add ONE nonclustered rowstore index for CaseNo lookups
CREATE NONCLUSTERED INDEX IX_PlannedServices_CaseNo
ON tbl_PlannedServices (CaseNo)
INCLUDE (Division,FiscalYear,STARTDATE,ENDDATE,IndexSubObjectCode,ServiceRatio,ConsumerCounty,
	GeographicDifferential,ProviderRateType,ServiceCode,Service,UnitType,UnitsPer,UnitsOfMeasure,TotalUnits,
	AnnualizedUnits,VendorID,ProviderName,ProviderMedcId,Rate,MaxAmount,COMMENTS,PlannedServiceStatus,
	RegionStateReviewComments,AllowEVVDelivery,EVVComments,DATETIMESTAMP,UserStamp,PlannedServiceId,
	PlanId,ISComboCodeID,VendorServicesId)

-- 3. Create unique non-clustered index for business key
CREATE UNIQUE NONCLUSTERED INDEX IX_PlannedServices_BusinessKey
ON tbl_PlannedServices (CaseNo,Division,FiscalYear,STARTDATE,ENDDATE,IndexSubObjectCode,ServiceRatio,ConsumerCounty,
	GeographicDifferential,ProviderRateType,ServiceCode,Service,VendorID,ProviderName,ProviderMedcId,Rate,MaxAmount,PlannedServiceStatus,
	DATETIMESTAMP,UserStamp,PlannedServiceId,PlanId,ISComboCodeID,VendorServicesId)

/*
-- By trial-error, the following query returned 1,066,576 records, 
-- which is ther size of the table. Therefore, this is the PK

SELECT COUNT(*) AS DistinctCombinations
FROM (
    SELECT DISTINCT PlannedServiceId
    FROM tbl_PlannedServices
) AS UniqueRows
*/

--================================================
-- [dbo].[tbl_Plans]
--================================================
-- 1. Primary 
ALTER TABLE tbl_Plans
ADD CONSTRAINT PK_Plans PRIMARY KEY CLUSTERED (PlanId)

-- 2. Add ONE nonclustered rowstore index for CaseNo lookups
CREATE NONCLUSTERED INDEX IX_Plans_CaseNo
ON tbl_Plans (CaseNo)
INCLUDE (Division,Program,Worker,CreationDate,Comments,
	[Status],BeginDate,EndDate,Review,ReviewRequestDate,UserStamp,DateTimeStamp,PlanId,BudgetId,OpenId,EnrollID)

-- 3. Create unique non-clustered index for business key
CREATE UNIQUE NONCLUSTERED INDEX IX_Planes_BusinessKey
ON tbl_Plans (CaseNo,Division,Program,Worker,CreationDate,[Status],BeginDate,EndDate,Review,ReviewRequestDate,UserStamp,DateTimeStamp,PlanId,BudgetId,OpenId,EnrollID)

/*
-- By trial-error, the following query returned 221,814 records, 
-- which is ther size of the table. Therefore, this is the PK

SELECT COUNT(*) AS DistinctCombinations
FROM (
    SELECT DISTINCT PlanId
    FROM tbl_Plans
) AS UniqueRows
*/


--================================================
-- [dbo].[tbl_QSIAssessments]
--================================================
-- 1. Primary 
ALTER TABLE tbl_QSIAssessments
ADD CONSTRAINT PK_QSIAssessments PRIMARY KEY CLUSTERED (AssessId)

-- 2. Add ONE nonclustered rowstore index for CaseNo lookups
CREATE NONCLUSTERED INDEX IX_QSIAssessments_CaseNo
ON tbl_QSIAssessments (CaseNo)
INCLUDE (ABCPIN,[STATUS],REVIEW,REVIEWDATE,RATER,RaterID,APPROVEDBY,APPROVEDATE,
	Q13a,Q13b,Q13c,Q14,Q15,Q16,Q17,Q18,Q19,Q20,Q21,Q22,Q23,Q24,Q25,Q26,Q27,Q28,
	Q29,Q30,Q31a,Q31b,Q32,Q33,Q34,Q35,Q36,Q37,Q38,Q39,Q40,Q41,Q42,Q43,Q44,Q45,
	Q46,Q47,Q48,Q49,Q50,Q51a,FLEVEL,BLEVEL,PLEVEL,OLEVEL,LOSRI,DATETIMESTAMP,
	UserStamp,AssessID,LegacyAssessID)

-- 3. Create unique non-clustered index for business key
-- The maximum limit for index key column list is 32.
CREATE UNIQUE NONCLUSTERED INDEX IX_QSIAssessments_BusinessKey
ON tbl_QSIAssessments (CaseNo,ABCPIN,[STATUS],REVIEW,REVIEWDATE,RATER,RaterID,APPROVEDBY,APPROVEDATE,
	FLEVEL,BLEVEL,PLEVEL,OLEVEL,LOSRI,DATETIMESTAMP,
	UserStamp,AssessID,LegacyAssessID)

/*
-- By trial-error, the following query returned 90,467 records, 
-- which is ther size of the table. Therefore, this is the PK

SELECT COUNT(*) AS DistinctCombinations
FROM (
    SELECT DISTINCT AssessID
    FROM tbl_QSIAssessments
) AS UniqueRows

*/



--================================================
-- [dbo].[tbl_QSIAssessmentsLegacy]
--================================================
-- Remove erroneous records with ABCPIN 0001015332 and 0000101434. ALl other values were null
delete from tbl_QSIAssessmentsLegacy where AssessID is null

-- Since there are NO NULL values in AssessID, simply alter the column
ALTER TABLE [dbo].[tbl_QSIAssessmentsLegacy]
ALTER COLUMN ASSESSID int NOT NULL;

-- 1. Primary 
ALTER TABLE tbl_QSIAssessmentsLegacy
ADD CONSTRAINT PK_QSIAssessmentsLegacy PRIMARY KEY CLUSTERED (AssessId)

-- 2. Add ONE nonclustered rowstore index for CaseNo lookups
CREATE NONCLUSTERED INDEX IX_QSIAssessmentsLegacy_CaseNo
ON tbl_QSIAssessmentsLegacy (ABCPIN)
INCLUDE ([STATUS],REVIEW,REVIEWDATE,RATER,RATERID,APPROVEDBY,CompletedDate,
	Q14,Q15,Q16,Q17,Q18,Q19,Q20,Q21,Q22,Q23,Q24,Q25,Q26,Q27,Q28,Q29,Q30,Q30a,
	Q30b,Q30bOther,Q31,Q32,Q33,Q34,Q35,Q36,Q37,Q38,Q39,Q40,Q41,Q42,Q43,Q43txt,
	Q44,Q45,Q46,Q47,Q48,Q49,Q49a,FLEVEL,BLEVEL,PLEVEL,OLEVEL,LOSRI,ASSESSID)

-- 3. Create unique non-clustered index for business key
-- The maximum limit for index key column list is 32.
CREATE UNIQUE NONCLUSTERED INDEX IX_QSIAssessmentsLegacy_BusinessKey
ON tbl_QSIAssessmentsLegacy (ABCPIN,[STATUS],REVIEW,REVIEWDATE,RATER,RATERID,APPROVEDBY,CompletedDate,
	FLEVEL,BLEVEL,PLEVEL,OLEVEL,LOSRI,ASSESSID)

/*
-- By trial-error, the following query returned 171,358 records, 
-- which is ther size of the table. Therefore, this is the PK

SELECT COUNT(*) AS DistinctCombinations
FROM (
    SELECT DISTINCT AssessID
    FROM tbl_QSIAssessmentsLegacy
) AS UniqueRows

*/


--================================================
-- [dbo].[tbl_QSIQuestions]
--================================================
-- Since there are NO NULL values in QuestionAssoc, simply alter the column
ALTER TABLE [dbo].tbl_QSIQuestions
ALTER COLUMN QuestionAssoc int NOT NULL;

-- 1. Primary 
ALTER TABLE tbl_QSIQuestions
ADD CONSTRAINT PK_QSIQuestions PRIMARY KEY CLUSTERED (QuestionID,QuestionAssoc)

-- 2. Add ONE nonclustered rowstore index for CaseNo lookups
CREATE NONCLUSTERED INDEX IX_QSIQuestions_QuestionID
ON tbl_QSIQuestions (QuestionID)
INCLUDE (Question,QuestionAssoc,QuestionAssocDescr,Descr)

-- 3. Create unique non-clustered index for business key
-- The maximum limit for index key column list is 32.
CREATE UNIQUE NONCLUSTERED INDEX IX_QSIQuestions_BusinessKey
ON tbl_QSIQuestions (QuestionID,Question,QuestionAssoc)

/*
-- By trial-error, the following query returned 198 records, 
-- which is ther size of the table. Therefore, this is the PK

SELECT COUNT(*) AS DistinctCombinations
FROM (
    SELECT DISTINCT QuestionID,QuestionAssoc
    FROM tbl_QSIQuestions
) AS UniqueRows
*/

--================================================
-- [dbo].[tbl_Rates]
--================================================

-- 1. Primary 
ALTER TABLE tbl_Rates
ADD CONSTRAINT PK_Rates PRIMARY KEY CLUSTERED (ServiceCodeUnitCostID)

-- 2. Add ONE nonclustered rowstore index for CaseNo lookups
CREATE NONCLUSTERED INDEX IX_Rates_ServiceCodeUnitCostID
ON tbl_Rates (ServiceCodeUnitCostID)
INCLUDE (ServiceCode,ServiceCodeiConnect,UnitCost,StartDate,EndDate,DateTimeStamp,AppType,
	FundCode,Credential,RateType,MaxUnits,Max1,Max2,UserStamp,BaseCost,ProviderRateType,InternalProgram,ConsumerCounty,ServiceRatio,ServiceCodesId)

-- 3. Create unique non-clustered index for business key
-- Warning! The maximum key length for a nonclustered index is 1700 bytes. The index 'IX_Rates_BusinessKey' has maximum length of 8909 bytes. 
-- For some combination of large values, the insert/update operation will fail.
CREATE UNIQUE NONCLUSTERED INDEX IX_Rates_BusinessKey
ON tbl_Rates (ServiceCodeUnitCostID, ServiceCode,ServiceCodeiConnect,UnitCost,StartDate,EndDate,DateTimeStamp,AppType,
	FundCode,Credential,RateType,MaxUnits,Max1,Max2,UserStamp,BaseCost,ProviderRateType,ServiceRatio,ServiceCodesId)

/*
-- By trial-error, the following query returned 2,656 records, 
-- which is ther size of the table. Therefore, this is the PK
SELECT COUNT(*) AS DistinctCombinations
FROM (
    SELECT DISTINCT ServiceCodeUnitCostID
    FROM tbl_Rates
) AS UniqueRows
*/

--================================================
-- [dbo].[tbl_SANs]
--================================================
-- 1. Primary storage as columnstore (best for aggregations)
CREATE CLUSTERED COLUMNSTORE INDEX CI_SANs
ON tbl_SANs

-- 2. Add ONE nonclustered rowstore index for CaseNo lookups
CREATE NONCLUSTERED INDEX IX_SANs_CaseNo
ON tbl_SANs (CaseNo)
INCLUDE (SanID,Division,[Type],SANDueToUpdatedAlgorithm,Reason,[Status],PlanID,WSC,StateOfficeReviewer,
	DateInitiated,SubmissionDate,RAIDate,DueDate,[60DaysDate],[30DaysDate],CurrentBudget,AlgorithmAmount,AmountUnAuthorized,
	BudgetSource,LastRefresh,WSCProposedBudget,WSCProposedProratedIncrease,WSCProposedAnnualizedBudget,
	WSCProposedAnnualizedIncrease,StateProposedProratedBudget,StateProposedProratedIncrease,
	StateProposedAnnualizedBudget,StateProposedAnnualizedIncrease,Recommendation,
	PersonMakingRecommd,RecommendationDate,Decision,Decisionby,DateNoticeSent,DateTimeStamp,UserStamp)

-- 3. Create unique non-clustered index for business key
CREATE UNIQUE NONCLUSTERED INDEX IX_SANs_BusinessKey
ON tbl_SANs (CaseNo, SanID,Division,[Type],SANDueToUpdatedAlgorithm,Reason,[Status],PlanID,WSC,StateOfficeReviewer,
	DateInitiated,SubmissionDate,RAIDate,[60DaysDate],[30DaysDate],CurrentBudget,AlgorithmAmount,AmountUnAuthorized,
	BudgetSource,WSCProposedBudget,WSCProposedProratedIncrease,WSCProposedAnnualizedBudget,
	WSCProposedAnnualizedIncrease,StateProposedProratedBudget,StateProposedProratedIncrease,
	StateProposedAnnualizedBudget,StateProposedAnnualizedIncrease,Recommendation,
	PersonMakingRecommd,RecommendationDate,Decision,Decisionby)

/*
-- By trial-error, the following query returned 44,750 records, 
--which is ther size of the table. Therefore, this is the PK

SELECT COUNT(*) AS DistinctCombinations
FROM (
    SELECT DISTINCT SanID
    FROM tbl_SANs
) AS UniqueRows
*/

--================================================
-- [dbo].[tbl_ServiceCodes]
--================================================

-- 1. Primary 
ALTER TABLE tbl_ServiceCodes
ADD CONSTRAINT PK_ServiceCodes PRIMARY KEY CLUSTERED (ServiceCode)

-- 2. Add ONE nonclustered rowstore index for CaseNo lookups
CREATE NONCLUSTERED INDEX IX_Rates_ServiceCode
ON tbl_ServiceCodes (ServiceCode)
INCLUDE (ServiceCodeiConnect,UnitType,Service,SecondaryCode,ServiceCategory,ServiceType,
	Active,EffectiveDate,InvoiceGroup,AuthRequ,AllowDuplicates,RequiresDiagnosis,AuthAllowed,
	AllowPartialUnits,HighAge,LowAge,TPLAction,MedicaidCovered,ServiceCodesId,MaxUnitLimit,UnitLimitFrequency,MaxAmountLimit,AmountLimitFrequency)

-- 3. Create unique non-clustered index for business key
-- Warning! The maximum key length for a nonclustered index is 1700 bytes. The index 'IX_Rates_BusinessKey' has maximum length of 8909 bytes. 
-- For some combination of large values, the insert/update operation will fail.
CREATE UNIQUE NONCLUSTERED INDEX IX_Rates_BusinessKey
ON tbl_ServiceCodes (ServiceCode,ServiceCodeiConnect,UnitType,Service,SecondaryCode,ServiceCategory,ServiceType,
	Active,EffectiveDate,InvoiceGroup,AuthRequ,AllowDuplicates,RequiresDiagnosis,AuthAllowed,
	AllowPartialUnits,HighAge,LowAge,TPLAction,MedicaidCovered,ServiceCodesId,MaxUnitLimit,UnitLimitFrequency,MaxAmountLimit,AmountLimitFrequency)


/*
-- By trial-error, the following query returned 373 records, 
-- which is ther size of the table. Therefore, this is the PK

SELECT COUNT(*) AS DistinctCombinations
FROM (
    SELECT DISTINCT ServiceCode
    FROM tbl_ServiceCodes
) AS UniqueRows
*/