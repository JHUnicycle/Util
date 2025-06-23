import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
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
        
        # 定义多元非线性模型
        self.multivariate_models = {
            'linear': {
                'func': lambda X, *params: np.sum([p * X[i] for i, p in enumerate(params[:-1])], axis=0) + params[-1],
                'name': '多元线性: y = a1*x1 + a2*x2 + ... + b',
                'init_guess': lambda X, y: np.ones(X.shape[0] + 1)
            },
            'quadratic_sum': {
                'func': lambda X, *params: np.sum([params[i] * X[i]**2 for i in range(X.shape[0])], axis=0) + 
                        np.sum([params[X.shape[0] + i] * X[i] for i in range(X.shape[0])], axis=0) + params[-1],
                'name': '二次和: y = a1*x1² + a2*x2² + ... + b1*x1 + b2*x2 + ... + c',
                'init_guess': lambda X, y: np.ones(2 * X.shape[0] + 1)
            },
            'interaction': {
                'func': lambda X, *params: (params[0] * X[0] + params[1] * X[1] + 
                         params[2] * X[0] * X[1] + params[3]) if X.shape[0] == 2 else None,
                'name': '交互作用: y = a1*x1 + a2*x2 + a3*x1*x2 + b',
                'init_guess': lambda X, y: [1, 1, 0.5, 0],
                'min_features': 2,
                'max_features': 2
            },
            'exponential_sum': {
                'func': lambda X, *params: np.sum([params[i] * np.exp(params[X.shape[0] + i] * X[i]) 
                         for i in range(X.shape[0])], axis=0) + params[-1],
                'name': '指数和: y = a1*exp(b1*x1) + a2*exp(b2*x2) + ... + c',
                'init_guess': lambda X, y: np.concatenate([np.ones(X.shape[0]), np.full(X.shape[0], 0.1), [0]])
            },
            'polynomial_degree2': {
                'func': lambda X, *params: self._polynomial_degree2(X, params),
                'name': '二次多项式: y = a1*x1 + a2*x2 + ... + b1*x1² + b2*x2² + ... + c1*x1*x2 + ... + d',
                'init_guess': lambda X, y: np.ones(X.shape[0] + X.shape[0] + X.shape[0]*(X.shape[0]-1)//2 + 1)
            },
            'logarithmic_sum': {
                'func': lambda X, *params: np.sum([params[i] * np.log(np.abs(X[i]) + 1e-10) 
                         for i in range(X.shape[0])], axis=0) + params[-1],
                'name': '对数和: y = a1*ln(x1) + a2*ln(x2) + ... + b',
                'init_guess': lambda X, y: np.ones(X.shape[0] + 1)
            },
            'power_sum': {
                'func': lambda X, *params: np.sum([params[i] * np.power(np.abs(X[i]) + 1e-10, params[X.shape[0] + i]) 
                         for i in range(X.shape[0])], axis=0) + params[-1],
                'name': '幂函数和: y = a1*x1^b1 + a2*x2^b2 + ... + c',
                'init_guess': lambda X, y: np.concatenate([np.ones(X.shape[0]), np.ones(X.shape[0]), [0]])
            }
        }
    
    def _polynomial_degree2(self, X, params):
        """计算二次多项式"""
        n_features = X.shape[0]
        result = params[-1]  # 常数项
        
        # 线性项
        for i in range(n_features):
            result += params[i] * X[i]
        
        # 二次项
        for i in range(n_features):
            result += params[n_features + i] * X[i]**2
        
        # 交互项
        idx = 2 * n_features
        for i in range(n_features):
            for j in range(i + 1, n_features):
                result += params[idx] * X[i] * X[j]
                idx += 1
        
        return result
    
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
    
    def format_multivariate_equation(self, model_name, params, column_names):
        """格式化多元拟合方程"""
        n_features = len(column_names) - 1
        
        if model_name == 'linear':
            terms = []
            for i in range(n_features):
                terms.append(f"{params[i]:.4f}*{column_names[i]}")
            terms.append(f"{params[-1]:.4f}")
            return f"{column_names[-1]} = " + " + ".join(terms)
            
        elif model_name == 'quadratic_sum':
            terms = []
            # 二次项
            for i in range(n_features):
                terms.append(f"{params[i]:.4f}*{column_names[i]}²")
            # 线性项
            for i in range(n_features):
                terms.append(f"{params[n_features + i]:.4f}*{column_names[i]}")
            terms.append(f"{params[-1]:.4f}")
            return f"{column_names[-1]} = " + " + ".join(terms)
            
        elif model_name == 'interaction' and n_features == 2:
            return f"{column_names[-1]} = {params[0]:.4f}*{column_names[0]} + {params[1]:.4f}*{column_names[1]} + {params[2]:.4f}*{column_names[0]}*{column_names[1]} + {params[3]:.4f}"
            
        elif model_name == 'exponential_sum':
            terms = []
            for i in range(n_features):
                terms.append(f"{params[i]:.4f}*exp({params[n_features + i]:.4f}*{column_names[i]})")
            terms.append(f"{params[-1]:.4f}")
            return f"{column_names[-1]} = " + " + ".join(terms)
            
        elif model_name == 'logarithmic_sum':
            terms = []
            for i in range(n_features):
                terms.append(f"{params[i]:.4f}*ln({column_names[i]})")
            terms.append(f"{params[-1]:.4f}")
            return f"{column_names[-1]} = " + " + ".join(terms)
            
        elif model_name == 'power_sum':
            terms = []
            for i in range(n_features):
                terms.append(f"{params[i]:.4f}*{column_names[i]}^{params[n_features + i]:.4f}")
            terms.append(f"{params[-1]:.4f}")
            return f"{column_names[-1]} = " + " + ".join(terms)
            
        elif model_name == 'polynomial_degree2':
            terms = []
            # 线性项
            for i in range(n_features):
                terms.append(f"{params[i]:.4f}*{column_names[i]}")
            # 二次项
            for i in range(n_features):
                terms.append(f"{params[n_features + i]:.4f}*{column_names[i]}²")
            # 交互项
            idx = 2 * n_features
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    terms.append(f"{params[idx]:.4f}*{column_names[i]}*{column_names[j]}")
                    idx += 1
            terms.append(f"{params[-1]:.4f}")
            return f"{column_names[-1]} = " + " + ".join(terms)
            
        else:
            return "未知多元模型"

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
        
        print("\n" + "="*50)
        print(f"最佳模型: {best_model['display_name']}")
        print(f"R² = {best_model['r2']:.4f}, RMSE = {best_model['rmse']:.4f}")
        print(f"最佳拟合方程: {self.format_equation(best_model['name'], best_model['params'])}")
        print("="*50)
        
        return best_model, all_results
    
    def auto_fit_multivariate(self, X, y, column_names, show_details=True):
        """自动选择最适合的多元模型"""
        print("开始多元函数自动模型选择...")
        all_results = []
        n_features = X.shape[1]
        
        for model_name, model_info in self.multivariate_models.items():
            try:
                # 检查特征数量限制
                if 'min_features' in model_info and n_features < model_info['min_features']:
                    if show_details:
                        print(f"\n{model_info['name']}: 跳过 - 需要至少{model_info['min_features']}个特征")
                    continue
                if 'max_features' in model_info and n_features > model_info['max_features']:
                    if show_details:
                        print(f"\n{model_info['name']}: 跳过 - 最多支持{model_info['max_features']}个特征")
                    continue
                
                # 获取初始猜测值
                init_guess = model_info['init_guess'](X.T, y)
                
                # 进行拟合
                params, covariance = curve_fit(
                    model_info['func'], X.T, y, 
                    p0=init_guess, maxfev=10000
                )
                
                # 计算预测值和质量指标
                y_pred = model_info['func'](X.T, *params)
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
                    print(f"  方程: {self.format_multivariate_equation(model_name, params, column_names)}")
                    
            except Exception as e:
                if show_details:
                    print(f"\n{model_info['name']}: 拟合失败 - {str(e)}")
        
        if not all_results:
            print("所有多元模型拟合均失败!")
            return None, []
        
        # 选择最佳模型（基于R²）
        best_model = max(all_results, key=lambda x: x['r2'])
        
        print("\n" + "="*60)
        print(f"最佳多元模型: {best_model['display_name']}")
        print(f"R² = {best_model['r2']:.4f}, RMSE = {best_model['rmse']:.4f}")
        print(f"最佳拟合方程: {self.format_multivariate_equation(best_model['name'], best_model['params'], column_names)}")
        print("="*60)
        
        return best_model, all_results

    def fit_1d(self, x, y, model_func=None):
        """一元函数拟合"""
        if model_func is None:
            def default_linear(x, a, b):
                return a * x + b
            model_func = default_linear
        params, covariance = curve_fit(model_func, x, y)
        return params, covariance
    
    def fit_2d(self, x, y, z, model_func=None):
        """二元函数拟合"""
        if model_func is None:
            def default_2d_linear(xy, a, b, c):
                return a * xy[0] + b * xy[1] + c
            model_func = default_2d_linear
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
    
    def plot_multivariate_comparison(self, all_results):
        """绘制多元模型比较图"""
        if len(all_results) <= 1:
            return
            
        plt.figure(figsize=(12, 8))
        
        # 左图：R²比较
        plt.subplot(2, 2, 1)
        model_names = [r['name'] for r in all_results]
        r2_scores = [r['r2'] for r in all_results]
        
        bars = plt.bar(range(len(model_names)), r2_scores)
        plt.xlabel('模型类型')
        plt.ylabel('R² 分数')
        plt.title('各多元模型R²比较')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # 高亮最佳模型
        best_idx = np.argmax(r2_scores)
        bars[best_idx].set_color('red')
        
        # 右图：RMSE比较
        plt.subplot(2, 2, 2)
        rmse_scores = [r['rmse'] for r in all_results]
        
        bars = plt.bar(range(len(model_names)), rmse_scores)
        plt.xlabel('模型类型')
        plt.ylabel('RMSE')
        plt.title('各多元模型RMSE比较')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # 高亮最佳模型（RMSE最小）
        best_rmse_idx = np.argmin(rmse_scores)
        bars[best_rmse_idx].set_color('blue')
        
        # 下图：AIC比较
        plt.subplot(2, 1, 2)
        aic_scores = [r['aic'] for r in all_results]
        
        bars = plt.bar(range(len(model_names)), aic_scores)
        plt.xlabel('模型类型')
        plt.ylabel('AIC')
        plt.title('各多元模型AIC比较（越小越好）')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # 高亮最佳模型（AIC最小）
        best_aic_idx = np.argmin(aic_scores)
        bars[best_aic_idx].set_color('green')
        
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
            
            if fit_data and auto_nonlinear:
                print(f"\n对 {column_names[-1]} vs {column_names[0]} & {column_names[1]} 进行多元自动模型选择...")
                best_model, all_results = self.auto_fit_multivariate(X, y, column_names)
                
                if best_model and all_results:
                    # 绘制3D散点图和最佳拟合面
                    fig = plt.figure(figsize=(15, 6))
                    
                    # 左图：3D拟合结果
                    ax1 = fig.add_subplot(121, projection='3d')
                    scatter = ax1.scatter(x1, x2, y, c=y, cmap='viridis', alpha=0.7, label='原始数据')
                    ax1.set_xlabel(column_names[0])
                    ax1.set_ylabel(column_names[1])
                    ax1.set_zlabel(column_names[-1])
                    ax1.set_title(f'最佳拟合: {best_model["display_name"]}\n(R² = {best_model["r2"]:.4f})')
                    
                    # 绘制拟合面
                    x1_grid, x2_grid = np.meshgrid(
                        np.linspace(x1.min(), x1.max(), 20),
                        np.linspace(x2.min(), x2.max(), 20)
                    )
                    xy_grid = np.vstack([x1_grid.flatten(), x2_grid.flatten()])
                    z_grid = best_model['func'](xy_grid, *best_model['params']).reshape(x1_grid.shape)
                    ax1.plot_surface(x1_grid, x2_grid, z_grid, alpha=0.3, color='red')
                    
                    plt.colorbar(scatter, ax=ax1, shrink=0.5, aspect=5)
                    
                    # 右图：模型比较
                    ax2 = fig.add_subplot(122)
                    if len(all_results) > 1:
                        model_names = [r['name'] for r in all_results]
                        r2_scores = [r['r2'] for r in all_results]
                        
                        bars = ax2.bar(range(len(model_names)), r2_scores)
                        ax2.set_xlabel('模型类型')
                        ax2.set_ylabel('R² 分数')
                        ax2.set_title('各多元模型拟合质量比较')
                        ax2.set_xticks(range(len(model_names)))
                        ax2.set_xticklabels(model_names, rotation=45, ha='right')
                        ax2.grid(True, alpha=0.3)
                        
                        # 高亮最佳模型
                        best_idx = model_names.index(best_model['name'])
                        bars[best_idx].set_color('red')
                    
                    plt.tight_layout()
                    plt.show()
                    
                    # 显示详细的模型比较图
                    self.plot_multivariate_comparison(all_results)
                    
            elif fit_data:
                # 传统二元线性拟合
                self.plot_2d_fit(x1, x2, y, column_names, fit_data)
            else:
                # 只显示原始数据
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(x1, x2, y, c=y, cmap='viridis', alpha=0.7, label='原始数据')
                ax.set_xlabel(column_names[0])
                ax.set_ylabel(column_names[1])
                ax.set_zlabel(column_names[-1])
                ax.set_title(f'{column_names[-1]} vs {column_names[0]} & {column_names[1]}')
                plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
                plt.show()
            
        else:
            # 多元函数（3维以上）
            if fit_data and auto_nonlinear:
                print(f"\n对 {n_features}元函数 {column_names[-1]} 进行自动模型选择...")
                best_model, all_results = self.auto_fit_multivariate(X, y, column_names)
                
                if best_model and all_results:
                    # 显示模型比较图
                    self.plot_multivariate_comparison(all_results)
                    
                    # 显示残差分析
                    y_pred = best_model['func'](X.T, *best_model['params'])
                    residuals = y - y_pred
                    
                    plt.figure(figsize=(15, 5))
                    
                    # 预测值 vs 实际值
                    plt.subplot(1, 3, 1)
                    plt.scatter(y, y_pred, alpha=0.7)
                    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
                    plt.xlabel('实际值')
                    plt.ylabel('预测值')
                    plt.title(f'预测 vs 实际\n(R² = {best_model["r2"]:.4f})')
                    plt.grid(True)
                    
                    # 残差分布
                    plt.subplot(1, 3, 2)
                    plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
                    plt.xlabel('残差')
                    plt.ylabel('频数')
                    plt.title('残差分布')
                    plt.grid(True)
                    
                    # 残差 vs 预测值
                    plt.subplot(1, 3, 3)
                    plt.scatter(y_pred, residuals, alpha=0.7)
                    plt.axhline(y=0, color='r', linestyle='--')
                    plt.xlabel('预测值')
                    plt.ylabel('残差')
                    plt.title('残差 vs 预测值')
                    plt.grid(True)
                    
                    plt.tight_layout()
                    plt.show()
                    
                # 绘制传统的多元数据可视化
                self.plot_multivariate(X, y, column_names, fit_data=False)
                
            else:
                # 传统多元拟合或只显示数据
                self.plot_multivariate(X, y, column_names, fit_data)

def main():
    fitter = LeastSquaresFitter()
    fitter.plot_csv_data('distance.csv', fit_data=True)

if __name__ == "__main__":
    main()