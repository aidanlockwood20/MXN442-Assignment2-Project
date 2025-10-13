import pandas as pd
import numpy as np

def load_data_with_deciles(csv_path, with_numerical_variables = False):
    df = pd.read_csv(csv_path)

    # Quantity deciles
    quantity_bins = [0, 48, 150, 300, 600, 1104, 2160, 4600, 10000, 25000]
    quantity_labels = [f'Decile {i}' for i in range(1, len(quantity_bins))]
    df['quantity_decile'] = pd.cut(df["quantity"], bins=quantity_bins, labels=quantity_labels, right=False, include_lowest=True)

    # Product bundling
    product_bundle_counts = df.groupby('tender_id')['lot_productCode'].nunique()
    df['product_bundle_count'] = df['tender_id'].map(product_bundle_counts)
    bundle_bins = [0, 235.3, 668.6, 1596.1, 2680.2, 3807.1, 4974.8, 6143.9, 7486.5, 9532.6]
    bundle_labels = [f'Decile {i}' for i in range(1, len(bundle_bins))]
    df['product_bundling_decile'] = pd.cut(df['product_bundle_count'], bins=bundle_bins, labels=bundle_labels, right=False, include_lowest=True)

    # Procedure type stays: translated_procedure_types
    
    # Submission period
    submission_bins = [0, 7, 15, 286]
    submission_labels = [f'Period {i}' for i in range(1, len(submission_bins))]
    df['submission_period_group'] = pd.cut(df['submission_period_days'], bins=submission_bins, labels=submission_labels, right=False, include_lowest=True)

    # Decision-making speed
    for col in ["tender_biddeadline", "tender_contractsignaturedate"]:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    df['decision_period_days'] = (
    df["tender_contractsignaturedate"] - df["tender_biddeadline"]
).dt.days
    decision_bins = [0, 190.5, 291.2, 468.4, 672.4, 895.9]
    decision_labels = [f'Decile {i}' for i in range(1, len(decision_bins))]
    df['decision_speed_decile'] = pd.cut(df['decision_period_days'], bins=decision_bins, labels=decision_labels, right=False, include_lowest=True)

    # Month from tender_publications_firstcallfor
    df['month'] = pd.to_datetime(df['tender_publications_firstcallfor'], errors='coerce').dt.month

    # Number of bidders
    bidder_bins = [0, 2, 5, 75]
    bidder_labels = [f'Decile {i}' for i in range(1, len(bidder_bins))]
    df['bidders_decile'] = pd.cut(df['tender_recordedbidscount'], bins=bidder_bins, labels=bidder_labels, right=False, include_lowest=True)

    # Market share deciles
    market_total = df['supplier_market_share'].sum()
    df['supplier_market_share_percent'] = 100 * df['supplier_market_share'] / market_total
    market_share_bins = [0, 0.036, 0.13, 0.37, 0.94, 2.75, 6.58, 10.99, 19.29, 36.37, 100]
    market_share_labels = [f'Decile {i}' for i in range(1, len(market_share_bins))]
    df['supplier_market_share_decile'] = pd.cut(df['supplier_market_share_percent'], 
                                            bins=market_share_bins, 
                                            labels=market_share_labels, 
                                            right=False, include_lowest=True)


    # Buyer's concentration deciles
    buyer_total_sum = df['buyer_annual_total'].sum()
    df['buyer_concentration_percent'] = 100 * df['buyer_annual_total'] / buyer_total_sum
    buyer_conc_bins = [0, 0.0018, 0.0058, 0.012, 0.019, 0.033, 0.053, 0.084, 0.10, 0.53, 100]
    buyer_conc_labels = [f'Decile {i}' for i in range(1, len(buyer_conc_bins))]
    df['buyer_concentration_decile'] = pd.cut(
        df['buyer_concentration_percent'],
        bins=buyer_conc_bins,
        labels=buyer_conc_labels,
        right=False,
        include_lowest=True
    )

    # Specialization??

    # Same location: bidder_city ?

    # Supplier size ??

    if with_numerical_variables:
        new_cols = [
            'quantity',
            'product_bundle_count',
            'translated_procedure_types',
            'submission_period_days',
            'decision_period_days',
            'month',
            'tender_recordedbidscount',
            'supplier_market_share_percent',
            'buyer_concentration_percent',
            'bidder_city',

            'lot_productCode',
            'tender_year',
            
            
            'log_unit_price'
        ]
    else:
        new_cols = [
            'quantity_decile',
            'product_bundling_decile',
            'translated_procedure_types',
            'submission_period_group',
            'decision_speed_decile',
            'month',
            'bidders_decile',
            'supplier_market_share_decile',
            'buyer_concentration_decile',
            'bidder_city',

            'lot_productCode',
            'tender_year',
            
            'log_unit_price'
        ]
    
    df_filtered = df[new_cols]
    return df_filtered