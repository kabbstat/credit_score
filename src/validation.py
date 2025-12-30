"""
Data validation module using schema-based validation.
Ensures data quality throughout the pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ColumnSchema:
    """Schema definition for a single column."""
    name: str
    dtype: Optional[str] = None
    nullable: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    max_null_percentage: float = 1.0  # 1.0 = 100% null allowed
    unique: bool = False


@dataclass
class DataSchema:
    """Schema definition for a dataset."""
    name: str
    columns: List[ColumnSchema] = field(default_factory=list)
    min_rows: int = 0
    max_rows: Optional[int] = None
    required_columns: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    check_name: str
    message: str
    details: Optional[Dict[str, Any]] = None


class DataValidator:
    """
    Data validator for ML pipeline.
    
    Validates data against a defined schema and custom rules.
    
    Example usage:
        validator = DataValidator()
        results = validator.validate(df)
        if not validator.is_valid(results):
            validator.raise_on_failure(results)
    """
    
    def __init__(self, schema: Optional[DataSchema] = None):
        """
        Initialize validator with optional schema.
        
        Args:
            schema: DataSchema defining validation rules
        """
        self.schema = schema or self._get_default_schema()
        self.results: List[ValidationResult] = []
    
    def _get_default_schema(self) -> DataSchema:
        """Get default schema for credit score data."""
        return DataSchema(
            name="credit_score_data",
            min_rows=1000,
            required_columns=[
                "customer_id",
                "annual_income",
                "outstanding_debt",
                "credit_utilization_ratio",
                "target"
            ],
            columns=[
                ColumnSchema(
                    name="annual_income",
                    dtype="float64",
                    min_value=0,
                    max_null_percentage=0.1
                ),
                ColumnSchema(
                    name="outstanding_debt",
                    dtype="float64",
                    min_value=0,
                    max_null_percentage=0.1
                ),
                ColumnSchema(
                    name="credit_utilization_ratio",
                    dtype="float64",
                    min_value=0,
                    max_value=100,
                    max_null_percentage=0.1
                ),
                ColumnSchema(
                    name="target",
                    nullable=False,
                    allowed_values=["Good", "Standard", "Poor"]
                )
            ]
        )
    
    def validate(self, df: pd.DataFrame) -> List[ValidationResult]:
        """
        Run all validation checks on dataframe.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            List of ValidationResult objects
        """
        self.results = []
        
        # Basic checks
        self.results.append(self._check_not_empty(df))
        self.results.append(self._check_min_rows(df))
        self.results.append(self._check_required_columns(df))
        
        # Column-level checks
        for col_schema in self.schema.columns:
            if col_schema.name in df.columns:
                self.results.extend(self._validate_column(df, col_schema))
        
        # Data quality checks
        self.results.append(self._check_duplicates(df))
        self.results.extend(self._check_data_types(df))
        
        # Log results
        self._log_results()
        
        return self.results
    
    def _check_not_empty(self, df: pd.DataFrame) -> ValidationResult:
        """Check that dataframe is not empty."""
        is_valid = len(df) > 0
        return ValidationResult(
            is_valid=is_valid,
            check_name="not_empty",
            message="DataFrame is not empty" if is_valid else "DataFrame is empty"
        )
    
    def _check_min_rows(self, df: pd.DataFrame) -> ValidationResult:
        """Check minimum row count."""
        is_valid = len(df) >= self.schema.min_rows
        return ValidationResult(
            is_valid=is_valid,
            check_name="min_rows",
            message=f"Row count ({len(df)}) >= minimum ({self.schema.min_rows})" if is_valid 
                    else f"Row count ({len(df)}) < minimum ({self.schema.min_rows})",
            details={"actual_rows": len(df), "min_required": self.schema.min_rows}
        )
    
    def _check_required_columns(self, df: pd.DataFrame) -> ValidationResult:
        """Check that all required columns are present."""
        missing = [col for col in self.schema.required_columns if col not in df.columns]
        is_valid = len(missing) == 0
        return ValidationResult(
            is_valid=is_valid,
            check_name="required_columns",
            message="All required columns present" if is_valid 
                    else f"Missing columns: {missing}",
            details={"missing_columns": missing}
        )
    
    def _validate_column(self, df: pd.DataFrame, col_schema: ColumnSchema) -> List[ValidationResult]:
        """Validate a single column against its schema."""
        results = []
        col = df[col_schema.name]
        
        # Null check
        null_pct = col.isna().mean()
        is_valid = null_pct <= col_schema.max_null_percentage
        results.append(ValidationResult(
            is_valid=is_valid,
            check_name=f"{col_schema.name}_null_percentage",
            message=f"{col_schema.name}: null percentage ({null_pct:.2%}) <= max ({col_schema.max_null_percentage:.2%})" if is_valid
                    else f"{col_schema.name}: null percentage ({null_pct:.2%}) > max ({col_schema.max_null_percentage:.2%})",
            details={"column": col_schema.name, "null_percentage": null_pct}
        ))
        
        # Min value check
        if col_schema.min_value is not None and pd.api.types.is_numeric_dtype(col):
            min_val = col.min()
            is_valid = pd.isna(min_val) or min_val >= col_schema.min_value
            results.append(ValidationResult(
                is_valid=is_valid,
                check_name=f"{col_schema.name}_min_value",
                message=f"{col_schema.name}: min value ({min_val}) >= {col_schema.min_value}" if is_valid
                        else f"{col_schema.name}: min value ({min_val}) < {col_schema.min_value}",
                details={"column": col_schema.name, "min_value": min_val}
            ))
        
        # Max value check
        if col_schema.max_value is not None and pd.api.types.is_numeric_dtype(col):
            max_val = col.max()
            is_valid = pd.isna(max_val) or max_val <= col_schema.max_value
            results.append(ValidationResult(
                is_valid=is_valid,
                check_name=f"{col_schema.name}_max_value",
                message=f"{col_schema.name}: max value ({max_val}) <= {col_schema.max_value}" if is_valid
                        else f"{col_schema.name}: max value ({max_val}) > {col_schema.max_value}",
                details={"column": col_schema.name, "max_value": max_val}
            ))
        
        # Allowed values check
        if col_schema.allowed_values is not None:
            unique_vals = set(col.dropna().unique())
            invalid_vals = unique_vals - set(col_schema.allowed_values)
            is_valid = len(invalid_vals) == 0
            results.append(ValidationResult(
                is_valid=is_valid,
                check_name=f"{col_schema.name}_allowed_values",
                message=f"{col_schema.name}: all values are allowed" if is_valid
                        else f"{col_schema.name}: invalid values found: {invalid_vals}",
                details={"column": col_schema.name, "invalid_values": list(invalid_vals)}
            ))
        
        return results
    
    def _check_duplicates(self, df: pd.DataFrame) -> ValidationResult:
        """Check for duplicate rows."""
        dup_count = df.duplicated().sum()
        dup_pct = dup_count / len(df) if len(df) > 0 else 0
        
        # Warning if >5% duplicates
        is_valid = dup_pct <= 0.05
        return ValidationResult(
            is_valid=is_valid,
            check_name="duplicates",
            message=f"Duplicate rows: {dup_count} ({dup_pct:.2%})",
            details={"duplicate_count": dup_count, "duplicate_percentage": dup_pct}
        )
    
    def _check_data_types(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Check data types match schema."""
        results = []
        
        for col_schema in self.schema.columns:
            if col_schema.name in df.columns and col_schema.dtype:
                actual_dtype = str(df[col_schema.name].dtype)
                is_valid = col_schema.dtype in actual_dtype or actual_dtype in col_schema.dtype
                results.append(ValidationResult(
                    is_valid=is_valid,
                    check_name=f"{col_schema.name}_dtype",
                    message=f"{col_schema.name}: dtype is {actual_dtype} (expected {col_schema.dtype})",
                    details={"column": col_schema.name, "actual": actual_dtype, "expected": col_schema.dtype}
                ))
        
        return results
    
    def _log_results(self):
        """Log validation results."""
        passed = sum(1 for r in self.results if r.is_valid)
        failed = len(self.results) - passed
        
        logger.info(f"Validation complete: {passed} passed, {failed} failed")
        
        for result in self.results:
            if result.is_valid:
                logger.debug(f"✓ {result.check_name}: {result.message}")
            else:
                logger.warning(f"✗ {result.check_name}: {result.message}")
    
    def is_valid(self, results: Optional[List[ValidationResult]] = None) -> bool:
        """Check if all validations passed."""
        results = results or self.results
        return all(r.is_valid for r in results)
    
    def get_failures(self, results: Optional[List[ValidationResult]] = None) -> List[ValidationResult]:
        """Get list of failed validations."""
        results = results or self.results
        return [r for r in results if not r.is_valid]
    
    def raise_on_failure(self, results: Optional[List[ValidationResult]] = None):
        """Raise exception if any validation failed."""
        failures = self.get_failures(results)
        if failures:
            messages = [f.message for f in failures]
            raise ValueError(f"Data validation failed:\n" + "\n".join(messages))
    
    def get_report(self, results: Optional[List[ValidationResult]] = None) -> Dict[str, Any]:
        """Generate validation report."""
        results = results or self.results
        
        return {
            "total_checks": len(results),
            "passed": sum(1 for r in results if r.is_valid),
            "failed": sum(1 for r in results if not r.is_valid),
            "is_valid": self.is_valid(results),
            "checks": [
                {
                    "name": r.check_name,
                    "is_valid": r.is_valid,
                    "message": r.message,
                    "details": r.details
                }
                for r in results
            ]
        }


def validate_raw_data(data_path: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate raw data file.
    
    Args:
        data_path: Path to raw data CSV
    
    Returns:
        Tuple of (is_valid, report)
    """
    logger.info(f"Validating raw data: {data_path}")
    
    df = pd.read_csv(data_path)
    validator = DataValidator()
    results = validator.validate(df)
    
    return validator.is_valid(results), validator.get_report(results)


def validate_processed_data(data_path: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate processed data file.
    
    Args:
        data_path: Path to processed data CSV
    
    Returns:
        Tuple of (is_valid, report)
    """
    logger.info(f"Validating processed data: {data_path}")
    
    # Schema for processed data (more strict)
    schema = DataSchema(
        name="processed_credit_score_data",
        min_rows=1000,
        required_columns=[
            "outstanding_debt",
            "annual_income",
            "credit_utilization_ratio",
            "target",
            "total_month"
        ],
        columns=[
            ColumnSchema(name="target", nullable=False),
            ColumnSchema(name="total_month", min_value=0),
            ColumnSchema(name="outstanding_debt", min_value=0),
            ColumnSchema(name="annual_income", min_value=0),
        ]
    )
    
    df = pd.read_csv(data_path)
    validator = DataValidator(schema=schema)
    results = validator.validate(df)
    
    return validator.is_valid(results), validator.get_report(results)


if __name__ == "__main__":
    # Test validation
    import json
    
    is_valid, report = validate_processed_data("data/processed/processed.csv")
    print(json.dumps(report, indent=2, default=str))
