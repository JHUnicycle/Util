import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['font.family'] = 'sans-serif'

class LeastSquaresFitter:
    """
    最小二乘法拟合类，支持一元、二元及多元函数拟合
    """
    def __init__(self):
        # 定义常见的一元非线性模型
        self.nonlinear_models = {
            'linear': {
                'func': lambda x, a, b: a * x + b,
                'name': '线性: y = ax + b',
                'init_guess': lambda x, y: [1, 0]
            },
            'quadratic': {
                'func': lambda x, a, b, c: a * x**2 + b * x + c,
                'name': '二次: y = ax² + bx + c',
                'init_guess': lambda x, y: [1, 1, 0]
            },
            'cubic': {
                'func': lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d,
                'name': '三次: y = ax³ + bx² + cx + d',
                'init_guess': lambda x, y: [1, 1, 1, 0]
            },
            'exponential': {
                'func': lambda x, a, b, c: a * np.exp(b * x) + c,
                'name': '指数: y = a·exp(bx) + c',
                'init_guess': lambda x, y: [1, 0.1, 0]
            },
            'logarithmic': {
                'func': lambda x, a, b: a * np.log(x + 1e-10) + b,
                'name': '对数: y = a·ln(x) + b',
                'init_guess': lambda x, y: [1, 0]
            },
            'power': {
                'func': lambda x, a, b, c: a * np.power(np.abs(x) + 1e-10, b) + c,
                'name': '幂函数: y = a·x^b + c',
                'init_guess': lambda x, y: [1, 1, 0]
            },
            'gaussian': {
                'func': lambda x, a, b, c, d: a * np.exp(-((x - b) / c)**2) + d,
                'name': '高斯: y = a·exp(-((x-b)/c)²) + d',
                'init_guess': lambda x, y: [np.max(y) - np.min(y), np.mean(x), np.std(x), np.min(y)]
            },
            'sine': {
                'func': lambda x, a, b, c, d: a * np.sin(b * x + c) + d,
                'name': '正弦: y = a·sin(bx + c) + d',
                'init_guess': lambda x, y: [np.std(y), 1, 0, np.mean(y)]
            },
            'sigmoid': {
                'func': lambda x, a, b, c, d: a / (1 + np.exp(-b * (x - c))) + d,
                'name': 'Sigmoid: y = a/(1+exp(-b(x-c))) + d',
                'init_guess': lambda x, y: [np.max(y) - np.min(y), 1, np.mean(x), np.min(y)]
            }
        }
    
    def calculate_fit_quality(self, y_true, y_pred):
        """计算拟合质量指标"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        n = len(y_true)
        aic = n * np.log(ss_res / n) + 2 * len(y_pred)
        return r2, rmse, aic
    
    def format_equation(self, model_name, params):
        """格式化拟合方程"""
        equations = {
            'linear': f"y = {params[0]:.6f}x + {params[1]:.6f}",
            'quadratic': f"y = {params[0]:.6f}x² + {params[1]:.6f}x + {params[2]:.6f}",
            'cubic': f"y = {params[0]:.6f}x³ + {params[1]:.6f}x² + {params[2]:.6f}x + {params[3]:.6f}",
            'exponential': f"y = {params[0]:.6f}·exp({params[1]:.6f}x) + {params[2]:.6f}",
            'logarithmic': f"y = {params[0]:.6f}·ln(x) + {params[1]:.6f}",
            'power': f"y = {params[0]:.6f}·x^{params[1]:.6f} + {params[2]:.6f}",
            'gaussian': f"y = {params[0]:.6f}·exp(-((x-{params[1]:.6f})/{params[2]:.6f})²) + {params[3]:.6f}",
            'sine': f"y = {params[0]:.6f}·sin({params[1]:.6f}x + {params[2]:.6f}) + {params[3]:.6f}",
            'sigmoid': f"y = {params[0]:.6f}/(1+exp(-{params[1]:.6f}(x-{params[2]:.6f}))) + {params[3]:.6f}"
        }
        return equations.get(model_name, "未知模型")
    
    def auto_fit_1d(self, x, y, show_details=True):
        """自动选择最适合的一元模型"""
        print("开始自动模型选择...")
        all_results = []
        
        for model_name, model_info in self.nonlinear_models.items():
            try:
                init_guess = model_info['init_guess'](x, y)
                params, covariance = curve_fit(
                    model_info['func'], x, y, 
                    p0=init_guess, maxfev=5000
                )
                
                y_pred = model_info['func'](x, *params)
                r2, rmse, aic = self.calculate_fit_quality(y, y_pred)
                
                result = {
                    'name': model_name,
                    'display_name': model_info['name'],
                    'params': params,
                    'func': model_info['func'],
                    'r2': r2,
                    'rmse': rmse,
                    'aic': aic
                }
                all_results.append(result)
                
                if show_details:
                    print(f"\n{model_info['name']}:")
                    print(f"  R² = {r2:.4f}, RMSE = {rmse:.4f}, AIC = {aic:.2f}")
                    print(f"  方程: {self.format_equation(model_name, params)}")
                    
            except Exception as e:
                if show_details:
                    print(f"\n{model_info['name']}: 拟合失败 - {str(e)}")
        
        if not all_results:
            print("所有模型拟合均失败!")
            return None, []
        
        best_model = max(all_results, key=lambda x: x['r2'])
        
        print(f"\n" + "="*50)
        print(f"最佳模型: {best_model['display_name']}")
        print(f"R² = {best_model['r2']:.4f}, RMSE = {best_model['rmse']:.4f}")
        print(f"最佳拟合方程: {self.format_equation(best_model['name'], best_model['params'])}")
        print("="*50)
        
        return best_model, all_results
    
    def fit_1d(self, x, y, model_func=None):
        """一元函数拟合"""
        if model_func is None:
            model_func = lambda x, a, b: a * x + b
        params, covariance = curve_fit(model_func, x, y)
        return params, covariance
    
    def fit_2d(self, x, y, z, model_func=None):
        """二元函数拟合"""
        if model_func is None:
            model_func = lambda xy, a, b, c: a * xy[0] + b * xy[1] + c
        xy = np.vstack((x.flatten(), y.flatten()))
        z = z.flatten()
        params, covariance = curve_fit(model_func, xy, z)
        return params, covariance
    
    def fit_nd(self, X, y):
        """多元线性拟合"""
        def linear_model(X, *params):
            return np.sum([p * X[i] for i, p in enumerate(params[:-1])], axis=0) + params[-1]
        
        n_params = X.shape[0] + 1
        initial_guess = np.ones(n_params)
        params, covariance = curve_fit(linear_model, X, y, p0=initial_guess)
        return params, covariance
    
    def read_csv_data(self, file_path):
        """从CSV文件读取数据"""
        try:
            data = pd.read_csv(file_path)
            print(f"成功读取CSV文件: {file_path}")
            print(f"数据形状: {data.shape}")
            print(f"列名: {list(data.columns)}")
            
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
            column_names = list(data.columns)
            
            print(f"自变量维度: {X.shape}")
            print(f"因变量维度: {y.shape}")
            
            return X, y, column_names
            
        except Exception as e:
            print(f"读取CSV文件时出错: {e}")
            return None, None, None
    
    def plot_1d_fit(self, x, y, best_model, all_results, column_names):
        """绘制一元拟合结果"""
        plt.figure(figsize=(15, 6))
        
        # 左图：最佳拟合结果
        plt.subplot(1, 2, 1)
        plt.scatter(x, y, alpha=0.7, label='原始数据')
        
        x_smooth = np.linspace(min(x), max(x), 200)
        y_smooth = best_model['func'](x_smooth, *best_model['params'])
        plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, 
                label=f"最佳拟合: {best_model['display_name']}")
        
        plt.xlabel(column_names[0])
        plt.ylabel(column_names[-1])
        plt.title(f'自动拟合结果 (R² = {best_model["r2"]:.4f})')
        plt.legend()
        plt.grid(True)
        
        # 右图：模型比较
        plt.subplot(1, 2, 2)
        if len(all_results) > 1:
            model_names = [r['name'] for r in all_results]
            r2_scores = [r['r2'] for r in all_results]
            
            bars = plt.bar(range(len(model_names)), r2_scores)
            plt.xlabel('模型类型')
            plt.ylabel('R² 分数')
            plt.title('各模型拟合质量比较')
            plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            # 高亮最佳模型
            best_idx = model_names.index(best_model['name'])
            bars[best_idx].set_color('red')
        
        plt.tight_layout()
        plt.show()
    
    def plot_2d_fit(self, x1, x2, y, column_names, fit_data=True):
        """绘制二元拟合结果"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x1, x2, y, c=y, cmap='viridis', alpha=0.7, label='原始数据')
        ax.set_xlabel(column_names[0])
        ax.set_ylabel(column_names[1])
        ax.set_zlabel(column_names[-1])
        ax.set_title(f'{column_names[-1]} vs {column_names[0]} & {column_names[1]}')
        
        plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        
        if fit_data:
            params, _ = self.fit_2d(x1, x2, y)
            
            x1_grid, x2_grid = np.meshgrid(
                np.linspace(x1.min(), x1.max(), 20),
                np.linspace(x2.min(), x2.max(), 20)
            )
            xy_grid = np.vstack([x1_grid.flatten(), x2_grid.flatten()])
            z_grid = (params[0] * xy_grid[0] + params[1] * xy_grid[1] + params[2]).reshape(x1_grid.shape)
            
            ax.plot_surface(x1_grid, x2_grid, z_grid, alpha=0.3, color='red')
            print(f"二元拟合结果: {column_names[-1]} = {params[0]:.3f} * {column_names[0]} + {params[1]:.3f} * {column_names[1]} + {params[2]:.3f}")
        
        plt.show()
    
    def plot_multivariate(self, X, y, column_names, fit_data=True):
        """绘制多元数据可视化"""
        n_features = X.shape[1]
        print(f"检测到{n_features}维自变量，绘制成对散点图矩阵...")
        
        # 成对散点图矩阵
        plot_data = np.column_stack([X, y])
        plot_df = pd.DataFrame(plot_data, columns=column_names)
        
        n_vars = len(column_names)
        fig, axes = plt.subplots(n_vars-1, n_vars-1, figsize=(15, 12))
        fig.suptitle('成对散点图矩阵', fontsize=16)
        
        for i in range(n_vars-1):
            for j in range(n_vars-1):
                if i == j:
                    axes[i, j].hist(plot_df.iloc[:, j], bins=20, alpha=0.7)
                    axes[i, j].set_title(f'{column_names[j]} 分布')
                else:
                    scatter = axes[i, j].scatter(plot_df.iloc[:, j], plot_df.iloc[:, i], 
                                               c=y, cmap='viridis', alpha=0.7)
                    axes[i, j].set_xlabel(column_names[j])
                    axes[i, j].set_ylabel(column_names[i])
                axes[i, j].grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=axes, shrink=0.8, label=column_names[-1])
        plt.tight_layout()
        plt.show()
        
        # 各自变量与因变量的关系
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_features):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            ax.scatter(X[:, i], y, alpha=0.7)
            ax.set_xlabel(column_names[i])
            ax.set_ylabel(column_names[-1])
            ax.set_title(f'{column_names[-1]} vs {column_names[i]}')
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_features, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        if fit_data:
            params, _ = self.fit_nd(X.T, y)
            equation_parts = []
            for i, param in enumerate(params[:-1]):
                equation_parts.append(f"{param:.3f} * {column_names[i]}")
            equation_parts.append(f"{params[-1]:.3f}")
            equation = f"{column_names[-1]} = " + " + ".join(equation_parts)
            print(f"多元拟合结果: {equation}")
    
    def plot_csv_data(self, file_path, fit_data=True, auto_nonlinear=True):
        """读取CSV数据并绘制图像，可选择是否进行拟合"""
        X, y, column_names = self.read_csv_data(file_path)
        
        if X is None or y is None:
            return
        
        n_features = X.shape[1]
        
        if n_features == 1:
            x = X[:, 0]
            
            if fit_data and auto_nonlinear:
                print(f"\n对 {column_names[-1]} vs {column_names[0]} 进行自动模型选择...")
                best_model, all_results = self.auto_fit_1d(x, y)
                
                if best_model:
                    self.plot_1d_fit(x, y, best_model, all_results, column_names)
                    
            elif fit_data:
                # 传统线性拟合
                plt.figure(figsize=(10, 6))
                plt.scatter(x, y, alpha=0.7, label='原始数据')
                plt.xlabel(column_names[0])
                plt.ylabel(column_names[-1])
                plt.title(f'{column_names[-1]} vs {column_names[0]}')
                plt.grid(True)
                
                params, _ = self.fit_1d(x, y)
                x_smooth = np.linspace(min(x), max(x), 100)
                y_fit = params[0] * x_smooth + params[1]
                plt.plot(x_smooth, y_fit, 'r-', label=f'线性拟合: y = {params[0]:.3f}x + {params[1]:.3f}')
                print(f"线性拟合结果: {column_names[-1]} = {params[0]:.3f} * {column_names[0]} + {params[1]:.3f}")
                
                plt.legend()
                plt.show()
            else:
                # 只显示原始数据
                plt.figure(figsize=(10, 6))
                plt.scatter(x, y, alpha=0.7, label='原始数据')
                plt.xlabel(column_names[0])
                plt.ylabel(column_names[-1])
                plt.title(f'{column_names[-1]} vs {column_names[0]}')
                plt.grid(True)
                plt.legend()
                plt.show()
                
        elif n_features == 2:
            x1, x2 = X[:, 0], X[:, 1]
            self.plot_2d_fit(x1, x2, y, column_names, fit_data)
            
        else:
            self.plot_multivariate(X, y, column_names, fit_data)

def main():
    fitter = LeastSquaresFitter()
    fitter.plot_csv_data('distance_y.csv', fit_data=True)

if __name__ == "__main__":
    main()