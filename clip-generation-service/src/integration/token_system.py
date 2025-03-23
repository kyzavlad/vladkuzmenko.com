import os
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
from datetime import datetime
import asyncio
from decimal import Decimal

@dataclass
class TokenConfig:
    """Configuration for token system integration."""
    base_cost_per_minute: Decimal = Decimal("2.0")
    face_tracking_cost_per_minute: Decimal = Decimal("1.0")
    highlight_detection_cost_per_minute: Decimal = Decimal("1.0")
    clip_generation_cost: Decimal = Decimal("0.5")
    bundle_discounts: Dict[int, Decimal] = None
    analytics_dir: str = "analytics"

class TokenSystemIntegrator:
    """Integrates with token system for usage tracking and billing."""
    
    def __init__(self, config: TokenConfig):
        """
        Initialize token system integrator.
        
        Args:
            config (TokenConfig): Token system configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize bundle discounts
        if config.bundle_discounts is None:
            config.bundle_discounts = {
                100: Decimal("0.95"),  # 5% discount for 100+ clips
                500: Decimal("0.90"),  # 10% discount for 500+ clips
                1000: Decimal("0.85")  # 15% discount for 1000+ clips
            }
        
        # Create analytics directory
        os.makedirs(config.analytics_dir, exist_ok=True)
    
    async def calculate_usage(
        self,
        video_duration: float,
        num_clips: int,
        features: List[str]
    ) -> Dict[str, Decimal]:
        """
        Calculate token usage for video processing.
        
        Args:
            video_duration (float): Video duration in minutes
            num_clips (int): Number of clips to generate
            features (List[str]): List of features to use
            
        Returns:
            Dict[str, Decimal]: Token usage breakdown
        """
        # Calculate base cost
        base_cost = self.config.base_cost_per_minute * Decimal(str(video_duration))
        
        # Calculate feature costs
        feature_costs = {}
        if "face_tracking" in features:
            feature_costs["face_tracking"] = (
                self.config.face_tracking_cost_per_minute * Decimal(str(video_duration))
            )
        
        if "highlight_detection" in features:
            feature_costs["highlight_detection"] = (
                self.config.highlight_detection_cost_per_minute * Decimal(str(video_duration))
            )
        
        # Calculate clip generation cost
        clip_cost = self.config.clip_generation_cost * Decimal(str(num_clips))
        
        # Calculate bundle discount
        discount = self._calculate_bundle_discount(num_clips)
        
        # Calculate total cost
        total_cost = base_cost + sum(feature_costs.values()) + clip_cost
        total_cost = total_cost * discount
        
        return {
            "base_cost": base_cost,
            "feature_costs": feature_costs,
            "clip_cost": clip_cost,
            "discount": discount,
            "total_cost": total_cost
        }
    
    def _calculate_bundle_discount(self, num_clips: int) -> Decimal:
        """
        Calculate bundle discount based on number of clips.
        
        Args:
            num_clips (int): Number of clips
            
        Returns:
            Decimal: Discount factor
        """
        # Find applicable discount
        for threshold, discount in sorted(self.config.bundle_discounts.items()):
            if num_clips >= threshold:
                return discount
        
        return Decimal("1.0")
    
    async def track_usage(
        self,
        user_id: str,
        usage: Dict[str, Decimal],
        metadata: Optional[Dict] = None
    ):
        """
        Track token usage.
        
        Args:
            user_id (str): User identifier
            usage (Dict[str, Decimal]): Usage breakdown
            metadata (Optional[Dict]): Additional metadata
        """
        # Create usage record
        record = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "usage": usage,
            "metadata": metadata or {}
        }
        
        # Save to analytics
        await self._save_usage_record(record)
        
        # Update user's token balance
        await self._update_token_balance(user_id, usage["total_cost"])
    
    async def _save_usage_record(self, record: Dict):
        """
        Save usage record to analytics.
        
        Args:
            record (Dict): Usage record
        """
        # Create filename with date
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"token_usage_{date_str}.json"
        filepath = os.path.join(self.config.analytics_dir, filename)
        
        # Load existing records
        records = []
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                records = json.load(f)
        
        # Add new record
        records.append(record)
        
        # Save updated records
        with open(filepath, "w") as f:
            json.dump(records, f, indent=2)
    
    async def _update_token_balance(
        self,
        user_id: str,
        amount: Decimal
    ):
        """
        Update user's token balance.
        
        Args:
            user_id (str): User identifier
            amount (Decimal): Amount to deduct
        """
        # TODO: Implement token balance update
        # This should integrate with the main token system
        pass
    
    async def generate_usage_report(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Generate usage report for user.
        
        Args:
            user_id (str): User identifier
            start_date (Optional[datetime]): Start date
            end_date (Optional[datetime]): End date
            
        Returns:
            Dict: Usage report
        """
        # Load usage records
        records = await self._load_usage_records(
            user_id,
            start_date,
            end_date
        )
        
        # Calculate totals
        total_usage = Decimal("0.0")
        feature_totals = {}
        
        for record in records:
            usage = record["usage"]
            total_usage += usage["total_cost"]
            
            for feature, cost in usage["feature_costs"].items():
                if feature not in feature_totals:
                    feature_totals[feature] = Decimal("0.0")
                feature_totals[feature] += cost
        
        return {
            "user_id": user_id,
            "period": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None
            },
            "total_usage": total_usage,
            "feature_totals": feature_totals,
            "num_records": len(records)
        }
    
    async def _load_usage_records(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Load usage records for user.
        
        Args:
            user_id (str): User identifier
            start_date (Optional[datetime]): Start date
            end_date (Optional[datetime]): End date
            
        Returns:
            List[Dict]: Usage records
        """
        records = []
        
        # Get date range
        if start_date is None:
            start_date = datetime.now().replace(day=1)  # First day of current month
        if end_date is None:
            end_date = datetime.now()
        
        # Load records from analytics directory
        for filename in os.listdir(self.config.analytics_dir):
            if not filename.startswith("token_usage_"):
                continue
            
            filepath = os.path.join(self.config.analytics_dir, filename)
            with open(filepath, "r") as f:
                file_records = json.load(f)
                
                for record in file_records:
                    if record["user_id"] != user_id:
                        continue
                    
                    record_date = datetime.fromisoformat(record["timestamp"])
                    if start_date <= record_date <= end_date:
                        records.append(record)
        
        return records

def main():
    """Main function for token system integration."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", type=str, required=True)
    parser.add_argument("--video_duration", type=float, required=True)
    parser.add_argument("--num_clips", type=int, required=True)
    parser.add_argument("--features", nargs="+", default=[])
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create token configuration
    config = TokenConfig()
    
    # Create integrator
    integrator = TokenSystemIntegrator(config)
    
    # Calculate usage
    usage = asyncio.run(integrator.calculate_usage(
        args.video_duration,
        args.num_clips,
        args.features
    ))
    
    # Track usage
    asyncio.run(integrator.track_usage(args.user_id, usage))
    
    # Generate report if dates provided
    if args.start_date or args.end_date:
        start_date = datetime.fromisoformat(args.start_date) if args.start_date else None
        end_date = datetime.fromisoformat(args.end_date) if args.end_date else None
        
        report = asyncio.run(integrator.generate_usage_report(
            args.user_id,
            start_date,
            end_date
        ))
        
        print("\nUsage Report:")
        print(json.dumps(report, indent=2, default=str))

if __name__ == "__main__":
    main() 