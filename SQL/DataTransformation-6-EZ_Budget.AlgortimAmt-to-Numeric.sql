-- SQL Server script to transform AlgorithmAmt column from varchar(50) to float
-- This script removes the $ sign and converts the column type

-- Step 1: Add a new temporary column with float type
-- Note: SQL Server uses 'float' instead of 'double'
ALTER TABLE tbl_EZBudget 
ADD AlgorithmAmt_temp float;

-- Step 2: Copy and convert the data from the original column
-- Remove the $ sign and convert to float
UPDATE tbl_EZBudget 
SET AlgorithmAmt_temp = CAST(REPLACE(AlgorithmAmt, '$', '') AS float)
WHERE AlgorithmAmt IS NOT NULL;

-- Step 3: Drop the original column
ALTER TABLE tbl_EZBudget 
DROP COLUMN AlgorithmAmt;

-- Step 4: Rename the temporary column to the original name
EXEC sp_rename 'tbl_EZBudget.AlgorithmAmt_temp', 'AlgorithmAmt', 'COLUMN';

-- Verify the changes
SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'tbl_EZBudget' AND COLUMN_NAME = 'AlgorithmAmt';

-- View sample data
SELECT TOP 10 AlgorithmAmt FROM tbl_EZBudget;