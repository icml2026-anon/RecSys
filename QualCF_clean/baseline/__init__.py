"""
Baseline models for comparison with QualCF
All models adapted to use RecBole framework for fair comparison
"""

from baseline.giffcf.giffcf import GiffCF
from baseline.dgcl.dgcl import DGCL
from baseline.cdiff4rec.cdiff4rec import CDiff4Rec

__all__ = ['GiffCF', 'DGCL', 'CDiff4Rec']
