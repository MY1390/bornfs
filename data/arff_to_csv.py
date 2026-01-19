#!/usr/bin/env python3
"""
ARFF形式のファイルをCSV形式に変換するスクリプト
スパース形式のARFFファイルに対応
"""

import re
import csv


def parse_arff_to_csv(arff_file, csv_file):
    """
    ARFF形式のファイルをCSV形式に変換

    Args:
        arff_file: 入力ARFFファイルのパス
        csv_file: 出力CSVファイルのパス
    """
    attributes = []
    data_rows = []
    in_data_section = False

    # ARFFファイルを読み込む
    with open(arff_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # 属性定義を解析
            if line.lower().startswith('@attribute'):
                parts = line.split()
                if len(parts) >= 2:
                    attr_name = parts[1]
                    attributes.append(attr_name)

            # データセクションの開始
            elif line.lower().startswith('@data'):
                in_data_section = True

            # データ行を解析（スパース形式）
            elif in_data_section and line and line.startswith('{') and line.endswith('}'):
                # スパース形式のデータを解析: {index value, index value, ...}
                row = ['0'] * len(attributes)  # デフォルト値は0

                # 中括弧を削除
                sparse_data = line[1:-1]

                # カンマで分割して各要素を処理
                pairs = sparse_data.split(',')
                for pair in pairs:
                    pair = pair.strip()
                    if pair:
                        # インデックスと値を分離
                        match = re.match(r'(\d+)\s+(\d+)', pair)
                        if match:
                            index = int(match.group(1))
                            value = match.group(2)
                            if index < len(row):
                                row[index] = value

                data_rows.append(row)

    # CSVファイルに書き込む
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # ヘッダー行を書き込む
        writer.writerow(attributes)

        # データ行を書き込む
        writer.writerows(data_rows)

    print(f"変換完了: {arff_file} -> {csv_file}")
    print(f"属性数: {len(attributes)}")
    print(f"データ行数: {len(data_rows)}")


if __name__ == '__main__':
    arff_file = 'test.arff'
    csv_file = 'test.csv'

    parse_arff_to_csv(arff_file, csv_file)
