import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches

class EvaluateProfit:
    def __init__(self):
        pass

    def build_profit_table_strategy_1(self, result_table, all_data, pv_range):
        profit_table = pd.DataFrame(columns=['in_date', 'in_price', 'out_date', 'out_price', 'pv', 'profit', 'profitability'])
        total_profit = 0
        for i in result_table.index:
            in_price = all_data['Open'].iloc[all_data.index.get_loc(result_table.loc[i, 't_date'])]
            out_price = all_data['Close'].iloc[all_data.index.get_loc(result_table.loc[i, 't_date'])+pv_range]
            if result_table.loc[i, 'pv'] == 'valley':
                profit = out_price - in_price
                profit_table.loc[i, 'pv'] = 'valley'
            else:
                profit = in_price - out_price
                profit_table.loc[i, 'pv'] = 'peak'
            profit_table.loc[i, 'in_date'] = result_table.loc[i, 't_date']
            profit_table.loc[i, 'in_price'] = in_price
            profit_table.loc[i, 'out_date'] = all_data.iloc[all_data.index.get_loc(result_table.loc[i, 't_date'])+pv_range].name
            profit_table.loc[i, 'out_price'] = out_price
            profit_table.loc[i, 'profit'] = profit
            profitability = round(profit/in_price, 4)
            profit_table.loc[i, 'profitability'] = f'{profitability} %'
            total_profit += profit
        return total_profit, profit_table

    def build_profit_table_strategy_2(self, result_table, all_data, profit_percentage, loss_percentage, pv_range):
        '''percentage:fioat
        '''
        profit_table = pd.DataFrame(columns=['in_date', 'in_price', 'out_date', 'out_price', 'pv', 'profit', 'strategy', 'strategy_price'])
        total_profit = 0
        for i in result_table.index:
            in_price = all_data['Open'].iloc[all_data.index.get_loc(result_table.loc[i, 't_date'])]
            if result_table.loc[i, 'pv'] == 'valley':
                stop_profit_price = in_price*(1+profit_percentage)
                stop_loss_price = in_price*(1-loss_percentage)
            else:
                stop_profit_price = in_price*(1-profit_percentage)
                stop_loss_price = in_price*(1+loss_percentage)

            # print(in_price, stop_loss_price, stop_profit_price)
            start_index = all_data.index.get_loc(result_table.loc[i, 't_date'])
            end_index = all_data.index.get_loc(result_table.loc[i, 't_date'])+pv_range
            trade_data = all_data.iloc[start_index:end_index]
            # check out_price
            out_price = all_data['Close'].iloc[end_index]
            profit_table.loc[i, 'out_date'] = all_data.iloc[end_index].name
            for j in trade_data.index:
                high = trade_data['High'].loc[j]
                low = trade_data['Low'].loc[j]
                if result_table.loc[i, 'pv'] == 'valley':
                    if high>=stop_profit_price:
                        out_price = stop_profit_price
                        profit_table.loc[i, 'out_date'] = trade_data.loc[j].name
                        profit_table.loc[i, 'strategy'] = 'stop_profit'
                        profit_table.loc[i, 'strategy_price'] = stop_profit_price
                        break
                    elif low<=stop_loss_price:
                        out_price = stop_loss_price
                        profit_table.loc[i, 'out_date'] = trade_data.loc[j].name
                        profit_table.loc[i, 'strategy'] = 'stop_loss'
                        profit_table.loc[i, 'strategy_price'] = stop_loss_price
                        break
                elif result_table.loc[i, 'pv'] == 'peak':
                        if low<=stop_profit_price:
                            out_price = stop_profit_price
                            profit_table.loc[i, 'out_date'] = trade_data.loc[j].name
                            profit_table.loc[i, 'strategy'] = 'stop_profit'
                            profit_table.loc[i, 'strategy_price'] = stop_profit_price
                            break
                        elif high>=stop_loss_price:
                            out_price = stop_loss_price
                            profit_table.loc[i, 'out_date'] = trade_data.loc[j].name
                            profit_table.loc[i, 'strategy'] = 'stop_loss'
                            profit_table.loc[i, 'strategy_price'] = stop_loss_price
                            break

            if result_table.loc[i, 'pv'] == 'valley':
                profit = out_price - in_price
                profit_table.loc[i, 'pv'] = 'valley'
            else:
                profit = in_price - out_price
                profit_table.loc[i, 'pv'] = 'peak'
            profit_table.loc[i, 'in_date'] = result_table.loc[i, 't_date']
            profit_table.loc[i, 'in_price'] = in_price
            
            profit_table.loc[i, 'out_price'] = out_price
            profit_table.loc[i, 'profit'] = profit
            profitability = round(profit/in_price, 4)
            profit_table.loc[i, 'profitability'] = f'{profitability} %'
            total_profit += profit
        return total_profit, profit_table

    def build_profit_table_strategy_3(self, result_table, all_data, profit_percentage, loss_percentage, pv_range):
        '''percentage:fioat
        '''
        profit_table = pd.DataFrame(columns=['in_date', 'in_price', 'out_date', 'out_price', 'pv', 'profit', 'strategy', 'strategy_price'])
        total_profit = 0
        for i in result_table.index:
            in_price = all_data['Open'].iloc[all_data.index.get_loc(result_table.loc[i, 't_date'])]
            # print(in_price, stop_loss_price, stop_profit_price)
            start_index = all_data.index.get_loc(result_table.loc[i, 't_date'])
            end_index = all_data.index.get_loc(result_table.loc[i, 't_date'])+pv_range
            trade_data = all_data.iloc[start_index:end_index]
            # check out_price
            out_price = all_data['Close'].iloc[end_index]
            profit_table.loc[i, 'out_date'] = all_data.iloc[end_index].name
            yesterday_close = all_data['Close'].iloc[all_data.index.get_loc(result_table.loc[i, 't_date'])-1]
            for j in trade_data.index:
                if result_table.loc[i, 'pv'] == 'valley':
                    stop_profit_price = yesterday_close*(1+profit_percentage)
                    stop_loss_price = yesterday_close*(1-loss_percentage)
                else:
                    stop_profit_price = yesterday_close*(1-profit_percentage)
                    stop_loss_price = yesterday_close*(1+loss_percentage)
                yesterday_close = trade_data['Close'].loc[j]
                high = trade_data['High'].loc[j]
                low = trade_data['Low'].loc[j]
                if result_table.loc[i, 'pv'] == 'valley':
                    if high>=stop_profit_price:
                        out_price = stop_profit_price
                        profit_table.loc[i, 'out_date'] = trade_data.loc[j].name
                        profit_table.loc[i, 'strategy'] = 'stop_profit'
                        profit_table.loc[i, 'strategy_price'] = stop_profit_price
                        break
                    elif low<=stop_loss_price:
                        out_price = stop_loss_price
                        profit_table.loc[i, 'out_date'] = trade_data.loc[j].name
                        profit_table.loc[i, 'strategy'] = 'stop_loss'
                        profit_table.loc[i, 'strategy_price'] = stop_loss_price
                        break
                elif result_table.loc[i, 'pv'] == 'peak':
                        if low<=stop_profit_price:
                            out_price = stop_profit_price
                            profit_table.loc[i, 'out_date'] = trade_data.loc[j].name
                            profit_table.loc[i, 'strategy'] = 'stop_profit'
                            profit_table.loc[i, 'strategy_price'] = stop_profit_price
                            break
                        elif high>=stop_loss_price:
                            out_price = stop_loss_price
                            profit_table.loc[i, 'out_date'] = trade_data.loc[j].name
                            profit_table.loc[i, 'strategy'] = 'stop_loss'
                            profit_table.loc[i, 'strategy_price'] = stop_loss_price
                            break

            if result_table.loc[i, 'pv'] == 'valley':
                profit = out_price - in_price
                profit_table.loc[i, 'pv'] = 'valley'
            else:
                profit = in_price - out_price
                profit_table.loc[i, 'pv'] = 'peak'
            profit_table.loc[i, 'in_date'] = result_table.loc[i, 't_date']
            profit_table.loc[i, 'in_price'] = in_price
            
            profit_table.loc[i, 'out_price'] = out_price
            profit_table.loc[i, 'profit'] = profit
            profitability = round(profit/in_price, 4)
            profit_table.loc[i, 'profitability'] = f'{profitability} %'
            total_profit += profit
        return total_profit, profit_table

    def _kbar(self, open, close, high, low, pos, ax):
        if close > open:             # 上漲
            color='green'                 # 紅 K 棒
            height=close - open   # 高度=收盤-開盤
            bottom=open             # 底部=開盤
        else:                               # 下跌
            color='red'              # 綠 k 棒
            height=open - close   # 高度=開盤-收盤
            bottom=close             # 底部=收盤
        # 繪製 k 棒實體      
        ax.bar(pos, height=height,bottom=bottom, width=1, color=color)
        # 繪製 k 棒上下影線
        ax.vlines(pos, high, low, color=color)

    def draw_profit_plot(self, profit_table, all_data):
        date_list = sorted(
            list(profit_table['in_date']) + list(profit_table['out_date']))
        plot_data = all_data.loc[date_list[0]:date_list[-1]]
        fig, ax = plt.subplots(figsize=(20, 8))
        for i in plot_data.index:
            self._kbar(plot_data['Open'].loc[i], plot_data['Close'].loc[i], plot_data['High'].loc[i], plot_data['Low'].loc[i], i, ax)
        for i in profit_table.index:
            if profit_table.loc[i, 'pv'] == 'peak':
                arrow = patches.FancyArrowPatch((profit_table.loc[i, 'in_date'], profit_table.loc[i, 'in_price']), (profit_table.loc[i, 'out_date'], profit_table.loc[i, 'out_price']), 
                                                linestyle='--', mutation_scale=20, arrowstyle='->', edgecolor='red')
                ax.add_patch(arrow)
                ax.plot(profit_table.loc[i, 'in_date'], plot_data['High'].loc[profit_table.loc[i, 'in_date']]+10, 'v', color='red')
                ax.plot(profit_table.loc[i, 'out_date'], plot_data['Low'].loc[profit_table.loc[i, 'out_date']]-10, '^', color='green')

            elif profit_table.loc[i, 'pv'] == 'valley':
                arrow = patches.FancyArrowPatch((profit_table.loc[i, 'in_date'], profit_table.loc[i, 'in_price']), (profit_table.loc[i, 'out_date'], profit_table.loc[i, 'out_price']), 
                                    linestyle='--', mutation_scale=20, arrowstyle='->', edgecolor='green')
                ax.add_patch(arrow)
                ax.plot(profit_table.loc[i, 'in_date'], plot_data['Low'].loc[profit_table.loc[i, 'in_date']]-10, '^', color='green')
                ax.plot(profit_table.loc[i, 'out_date'], plot_data['High'].loc[profit_table.loc[i, 'out_date']]+10, 'v', color='red')
                
            if 'strategy' in profit_table.columns:
                if profit_table.loc[i, 'strategy'] == 'stop_profit':
                    ax.hlines(profit_table.loc[i, 'strategy_price'], profit_table.loc[i, 'in_date'], profit_table.loc[i, 'out_date'], color='green', linestyles='dotted')
                elif profit_table.loc[i, 'strategy'] == 'stop_loss':
                    ax.hlines(profit_table.loc[i, 'strategy_price'], profit_table.loc[i, 'in_date'], profit_table.loc[i, 'out_date'], color='red', linestyles='dotted')
                
        # ax.annotate(f'{i}', (profit_table.loc[i, 'in_date'], profit_table.loc[i, 'in_price']-100), fontsize=14, c='black')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.grid()
        # plt.legend()
        plt.show()