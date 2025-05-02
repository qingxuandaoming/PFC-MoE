from manim import *
import numpy as np
from typing import List


class ImprovedMoEComparison(Scene):
    def construct(self):
        # 初始化配置参数
        self.setup_parameters()
        
        # 创建网络结构
        self.create_traditional_nn()
        self.create_improved_moe_nn()
        
        # 展示传统神经网络
        self.show_traditional_nn()
        
        # 转换为MoE结构
        self.transform_to_moe()
        
        # 演示动态路由
        self.demo_dynamic_routing()
        
        # 展示结论
        self.show_conclusion()
        
    def setup_parameters(self):
        """设置动画参数和颜色配置"""
        # 网络结构参数
        self.neuron_radius = 0.18
        self.layer_sizes = [[4], [5], [3]]  # 传统网络结构
        self.expert_sizes = [4]  # 专家网络规模
        self.num_experts = 3  # 专家数量
        
        # 动画参数 - 放慢节奏以便更好地理解
        self.default_fade_time = 1.2
        self.default_transform_time = 1.8
        self.default_wait_time = 1.5
        
        # 字幕系统参数 - 调整位置更靠下并支持双语
        self.subtitle_font_size = 28
        self.subtitle_color = WHITE
        self.subtitle_buff = 0.8  # 减小字幕与底部的距离，使其更靠下
        self.subtitle_en_font_size = 22  # 英文字幕字体大小
        
        # 颜色配置 - 使用更鲜艳的颜色
        self.input_color = BLUE_D
        self.hidden_color = GRAY
        self.output_color = GREEN_D
        self.gate_color = YELLOW_D
        self.expert_colors = [PINK, ORANGE, PURPLE]
        self.highlight_color = RED_A
        self.text_color = WHITE
        
        # 初始化MoE方程组 - 提前创建以避免重叠
        self.moe_equations = VGroup(
            MathTex(r"y = \sigma(W^{(L)} \cdots \sigma(W^{(1)}x + b^{(1)}) \cdots + b^{(L)})", font_size=24, color=BLUE),
            MathTex(r"y = \sum_{i=1}^n \text{Softmax}(W_g x + b_g)_i E_i(x)", font_size=24, color=YELLOW)
        ).arrange(DOWN, buff=1)

    def create_traditional_nn(self):
        """创建传统神经网络结构"""
        # 创建神经元层
        layers = VGroup()
        for i, size in enumerate(self.layer_sizes):
            # 为不同层设置不同颜色
            if i == 0:
                color = self.input_color
            elif i == len(self.layer_sizes) - 1:
                color = self.output_color
            else:
                color = self.hidden_color
                
            layer = VGroup(*[
                self.create_neuron(color)
                for _ in range(size[0])
            ]).arrange(DOWN, buff=0.5)
            layers.add(layer)

        layers.arrange(RIGHT, buff=1.8)
        self.nn_layers = layers

        # 创建连接线
        self.connections = VGroup()
        for i in range(len(layers) - 1):
            for src in layers[i]:
                for dst in layers[i + 1]:
                    line = self.create_connection(src, dst, self.hidden_color, 1.5)
                    self.connections.add(line)

        # 添加层标签
        layer_labels = VGroup(
            Text("Input Layer", font_size=20, color=self.text_color).next_to(layers[0], DOWN),
            Text("Hidden Layer", font_size=20, color=self.text_color).next_to(layers[1], DOWN),
            Text("Output Layer", font_size=20, color=self.text_color).next_to(layers[2], DOWN)
        )
        
        self.traditional_group = VGroup(layers, self.connections, layer_labels)

    def create_neuron(self, color=GRAY, radius=None):
        """创建神经元"""
        if radius is None:
            radius = self.neuron_radius
            
        neuron = Circle(radius=radius, color=color, fill_opacity=0.5)
        neuron.set_stroke(color, width=2)
        return neuron
    
    def create_connection(self, src, dst, color=GRAY, width=1.5):
        """创建连接线 - 确保连接严丝合缝"""
        # 精确计算起点和终点，确保连接线严丝合缝
        # 使用get_boundary_point方法获取神经元边界上的精确点
        start_point = src.get_boundary_point(dst.get_center() - src.get_center())
        end_point = dst.get_boundary_point(src.get_center() - dst.get_center())
        
        line = Line(
            start_point, 
            end_point,
            color=color, 
            stroke_width=width
        )
        return line
        
    def create_improved_moe_nn(self):
        """创建改进的MoE结构"""
        # 输入输出层 - 调整垂直间距以避免遮挡
        input_layer = VGroup(*[self.create_neuron(self.input_color) for _ in range(self.layer_sizes[0][0])])
        input_layer.arrange(DOWN, buff=0.6)  # 增加垂直间距
        
        output_layer = VGroup(*[self.create_neuron(self.output_color) for _ in range(self.layer_sizes[-1][0])])
        output_layer.arrange(DOWN, buff=0.6)  # 增加垂直间距

        # 门控网络（垂直布局）- 添加更多视觉细节并优化布局
        gate_layer = VGroup(*[
            VGroup(
                self.create_neuron(self.gate_color, radius=self.neuron_radius*1.1),
                Text(f"G{i+1}", font_size=16, color=BLACK).scale(0.7)
            )
            for i in range(self.num_experts)
        ]).arrange(DOWN, buff=0.7)  # 增加垂直间距
        gate_layer.next_to(input_layer, RIGHT, buff=2.0)  # 增加水平间距

        # 专家层（水平布局）- 添加专家标识并优化布局
        experts = VGroup()
        for i in range(self.num_experts):
            expert_neurons = VGroup(*[
                self.create_neuron(self.expert_colors[i])
                for _ in range(self.expert_sizes[0])
            ]).arrange(DOWN, buff=0.5)  # 增加垂直间距
            
            # 添加专家标识
            expert_label = Text(f"Expert {i+1}", font_size=16, color=self.expert_colors[i])
            expert_label.next_to(expert_neurons, UP, buff=0.3)  # 增加标签间距
            
            expert_group = VGroup(expert_neurons, expert_label)
            experts.add(expert_group)
            
        # 调整专家组的位置，避免与门控网络重叠
        experts.arrange(RIGHT, buff=1.8).shift(DOWN * 1.0 + RIGHT * 0.8)

        # 输出层位置调整
        output_layer.next_to(experts, RIGHT, buff=2.0)  # 增加水平间距

        # 连接线 - 优化连接线，使其更加清晰有序
        gate_conn = VGroup()
        expert_conn = VGroup()

        # 输入到门控连接 - 使用更有序的连接方式
        for i, src in enumerate(input_layer):
            # 为每个输入神经元只连接到对应的门控神经元，减少交叉
            gate_idx = i % len(gate_layer)
            gate = gate_layer[gate_idx]
            
            # 创建更清晰的虚线连接
            line = DashedLine(
                src.get_boundary_point(gate[0].get_center() - src.get_center()),
                gate[0].get_boundary_point(src.get_center() - gate[0].get_center()),
                color=self.gate_color, 
                stroke_width=1.5,
                dash_length=0.1
            )
            gate_conn.add(line)

        # 门控到专家连接 - 使用更清晰的连接
        for i, gate in enumerate(gate_layer):
            expert_group = experts[i]
            expert_neurons = expert_group[0]  # 获取专家神经元组
            
            # 创建从门控到专家的主连接线
            line = Line(
                gate[0].get_boundary_point(expert_neurons.get_center() - gate[0].get_center()),
                expert_neurons[0].get_boundary_point(gate[0].get_center() - expert_neurons[0].get_center()),
                color=self.gate_color, 
                stroke_width=4
            )
            expert_conn.add(line)

            # 暂时不添加专家到输出的连接，将在moe_group创建后处理

        expert_conn.add(line)

        # 标签系统 - 使用更清晰的标签
        labels = VGroup(
            Text("Input", font_size=24, color=self.text_color).next_to(input_layer, UP),
            Text("Gating Network", font_size=24, color=self.text_color).next_to(gate_layer, UP),
            Text("Mixture of Experts", font_size=24, color=self.text_color).next_to(experts, UP, buff=0.5),
            Text("Output", font_size=24, color=self.text_color).next_to(output_layer, UP)
        )

        self.moe_group = VGroup(
            input_layer, gate_layer, experts, output_layer,
            gate_conn, expert_conn, labels
        ).scale(0.9)
        
        # 创建专家到输出的连接 - 在moe_group初始化后添加
        expert_output_conn = VGroup()
        for i, expert_group in enumerate(experts):
            expert_neurons = expert_group[0]  # 获取专家神经元组
            
            # 为每个专家创建连接线到输出层
            for j, src in enumerate(expert_neurons):
                # 为每个专家神经元只连接到对应的输出神经元，减少交叉
                out_idx = j % len(output_layer)
                dst = output_layer[out_idx]
                
                # 使用贝塞尔曲线创建更平滑的连接
                # 计算控制点以避免线条交叉
                start_point = src.get_boundary_point(dst.get_center() - src.get_center())
                end_point = dst.get_boundary_point(src.get_center() - dst.get_center())
                
                # 添加垂直偏移以避免路径重叠
                vertical_offset = (j - len(expert_neurons)/2) * 0.05
                
                # 创建控制点
                control_point1 = start_point + RIGHT * 0.5 + UP * vertical_offset
                control_point2 = end_point + LEFT * 0.5 + UP * vertical_offset
                
                # 创建贝塞尔曲线
                curve = CubicBezier(
                    start_point,
                    control_point1,
                    control_point2,
                    end_point,
                    color=self.expert_colors[i],
                    stroke_width=1.5
                )
                
                expert_output_conn.add(curve)
        
        # 将专家到输出的连接添加到moe_group中
        self.moe_group.add(expert_output_conn)

    def show_subtitle(self, text, duration=2.0, position="bottom", en_text=None):
        """显示中英双语字幕，优化防止溢出屏幕"""
        # 创建中文字幕
        subtitle_cn = Text(text, font_size=self.subtitle_font_size, color=self.subtitle_color)
        
        # 检查字幕长度，如果太长则自动调整字体大小
        if len(text) > 30:
            subtitle_cn = Text(text, font_size=max(self.subtitle_font_size - 4, 20), color=self.subtitle_color)
        
        # 如果提供了英文字幕，创建英文字幕并放在中文下方
        if en_text:
            # 检查英文字幕长度，如果太长则自动调整字体大小
            en_font_size = self.subtitle_en_font_size
            if len(en_text) > 40:
                en_font_size = max(self.subtitle_en_font_size - 4, 16)
                
            subtitle_en = Text(en_text, font_size=en_font_size, color=self.subtitle_color)
            subtitle_group = VGroup(subtitle_cn, subtitle_en).arrange(DOWN, buff=0.2)
            
            # 确保字幕不会太宽，超过屏幕宽度的80%时进行缩放
            if subtitle_group.width > config.frame_width * 0.8:
                scale_factor = config.frame_width * 0.8 / subtitle_group.width
                subtitle_group.scale(scale_factor)
        else:
            subtitle_group = subtitle_cn
            # 确保单行字幕不会太宽
            if subtitle_group.width > config.frame_width * 0.8:
                scale_factor = config.frame_width * 0.8 / subtitle_group.width
                subtitle_group.scale(scale_factor)
        
        # 调整字幕位置，避免遮挡动画内容
        if position == "bottom":
            subtitle_group.to_edge(DOWN, buff=self.subtitle_buff)
        elif position == "top":
            subtitle_group.to_edge(UP, buff=self.subtitle_buff)
        
        # 添加半透明背景，提高字幕可读性
        subtitle_bg = BackgroundRectangle(subtitle_group, color=BLACK, fill_opacity=0.3)
        subtitle_with_bg = VGroup(subtitle_bg, subtitle_group)
        
        self.play(FadeIn(subtitle_with_bg), run_time=0.5)
        self.wait(duration)
        self.play(FadeOut(subtitle_with_bg), run_time=0.5)
        return subtitle_with_bg
        
    def show_traditional_nn(self):
        """展示传统神经网络"""
        # 清理场景，确保没有遗留元素
        self.clear()
        
        # 添加标题和说明
        title = Text("传统神经网络", font_size=36, color=self.text_color).to_edge(UP)
        subtitle = Text("所有神经元之间的密集连接", font_size=24, color=self.text_color)
        subtitle.next_to(title, DOWN, buff=0.3)
        
        # 创建背景框 - 调整透明度
        background = BackgroundRectangle(self.traditional_group, color=BLACK, fill_opacity=0.08)
        
        # 动画展示
        self.play(FadeIn(background))
        self.play(Write(title), Write(subtitle), run_time=self.default_transform_time)
        
        # 添加字幕解释
        self.show_subtitle("传统神经网络使用密集连接，每个神经元都与下一层的所有神经元相连", 
                          duration=3.0, 
                          en_text="Traditional neural networks use dense connections, with each neuron connected to all neurons in the next layer")
        
        # 分层展示网络结构 - 使用层标签字典跟踪已创建的标签
        layer_labels = {}
        
        for i, layer in enumerate(self.nn_layers):
            layer_name = ["Input", "Hidden", "Output"][i]
            layer_desc = Text(f"{layer_name} Layer", font_size=20, color=self.text_color)
            
            # 调整标签位置，避免与上方文字重叠
            if i == 1:  # Hidden Layer
                layer_desc.next_to(layer, UP, buff=0.8)  # 增加与层的间距，避免与上方文字重叠
            else:
                layer_desc.next_to(layer, UP, buff=0.4)  # 保持其他标签的原有间距
                
            layer_labels[i] = layer_desc
            
            self.play(
                FadeIn(layer, shift=UP),
                Write(layer_desc),
                run_time=self.default_transform_time
            )
            
            # 添加层级解释字幕
            layer_explanations = [
                "输入层接收原始数据，如图像像素或文本特征",
                "隐藏层提取和转换特征，学习数据的抽象表示",
                "输出层产生最终预测结果，如分类概率或回归值"
            ]
            
            layer_explanations_en = [
                "Input layer receives raw data, such as image pixels or text features",
                "Hidden layer extracts and transforms features, learning abstract representations of data",
                "Output layer produces final predictions, such as classification probabilities or regression values"
            ]
            
            if i < len(layer_explanations):
                self.show_subtitle(layer_explanations[i], duration=2.5, en_text=layer_explanations_en[i])
            
            if i > 0:  # 添加连接线
                conn_group = VGroup()
                for conn in self.connections:
                    # 精确匹配连接线的起点和终点
                    if conn.get_start()[0] == self.nn_layers[i-1][0].get_center()[0] and \
                       conn.get_end()[0] == layer[0].get_center()[0]:
                        # 重新计算连接点以确保严丝合缝
                        src_layer = self.nn_layers[i-1]
                        dst_layer = layer
                        for src_neuron in src_layer:
                            for dst_neuron in dst_layer:
                                # 找到对应的连接线并更新
                                if np.allclose(conn.get_start(), src_neuron.get_center() + RIGHT * src_neuron.radius) and \
                                   np.allclose(conn.get_end(), dst_neuron.get_center() + LEFT * dst_neuron.radius):
                                    # 使用更精确的边界点
                                    start = src_neuron.get_boundary_point(dst_neuron.get_center() - src_neuron.get_center())
                                    end = dst_neuron.get_boundary_point(src_neuron.get_center() - dst_neuron.get_center())
                                    conn.put_start_and_end_on(start, end)
                        conn_group.add(conn)
                
                self.play(Create(conn_group), run_time=self.default_transform_time)
                
                # 添加连接解释字幕
                if i == 1:
                    self.show_subtitle("每个神经元都与下一层的所有神经元相连，形成密集连接", 
                                      duration=2.5, 
                                      en_text="Each neuron is connected to all neurons in the next layer, forming dense connections")
                elif i == 2:
                    self.show_subtitle("这种全连接结构在处理复杂任务时计算成本高", 
                                      duration=2.5, 
                                      en_text="This fully connected structure has high computational cost when processing complex tasks")
        
        self.wait(0.8)

        # 展示数据流动 - 使用多个数据点和更平滑的路径
        self.demonstrate_data_flow(self.nn_layers)
        
        self.traditional_title = VGroup(title, subtitle)

    def demonstrate_data_flow(self, layers, paths=None, color=BLUE, num_points=3):
        """展示数据在网络中的流动 - 优化路径避免重叠"""
        if paths is None:
            # 创建默认路径 - 从输入到输出的直接路径，避免路径重叠
            paths = []
            for i in range(min(num_points, len(layers[0]))):
                path = VMobject()
                start = layers[0][i].get_center()
                end = layers[-1][i % len(layers[-1])].get_center()
                
                # 创建平滑路径 - 为每条路径添加垂直偏移，避免重叠
                vertical_offset = (i - (num_points-1)/2) * 0.1  # 根据路径索引计算垂直偏移
                
                control_points = []
                for j in range(len(layers)):
                    if j == 0:
                        control_points.append(start)
                    elif j == len(layers) - 1:
                        control_points.append(end)
                    else:
                        # 添加垂直偏移到中间控制点
                        mid_point = layers[j][i % len(layers[j])].get_center() + UP * vertical_offset
                        # 增加控制点间距，使曲线更平滑
                        control_points.extend([mid_point - RIGHT*0.6, mid_point, mid_point + RIGHT*0.6])
                
                path.set_points_smoothly(control_points)
                paths.append(path)
        
        # 创建数据点 - 使用不同颜色区分不同路径
        colors = [BLUE, BLUE_B, BLUE_C] if num_points >= 3 else [BLUE] * num_points
        data_points = VGroup(*[Dot(color=colors[i], radius=0.12, fill_opacity=0.8) for i in range(len(paths))])
        
        # 添加数据流动解释字幕
        self.show_subtitle("数据在网络中从输入层流向输出层，经过每一层的变换", 
                          duration=2.0, 
                          en_text="Data flows from input to output layer, transformed at each layer")
        
        # 动画展示数据流动 - 调整时间和间隔
        animations = []
        for i, (path, point) in enumerate(zip(paths, data_points)):
            animations.append(MoveAlongPath(
                point, path,
                run_time=3.5,  # 增加运行时间以便更好观察
                rate_func=smooth,
                remover=True
            ))
        
        self.play(LaggedStart(*animations, lag_ratio=0.3))  # 增加延迟比例
        
        # 添加总结字幕
        self.show_subtitle("传统神经网络中的每个神经元都参与计算，无论输入数据的特性如何", 
                          duration=2.5, 
                          en_text="In traditional neural networks, every neuron participates in computation regardless of input data characteristics")
        
    def transform_to_moe(self):
        """转换为MoE结构"""
        # 清理场景，确保没有遗留元素
        self.play(*[FadeOut(obj) for obj in self.mobjects if obj not in [self.nn_layers[0], self.nn_layers[2], self.traditional_title]])
        
        # 新标题和说明
        new_title = Text("混合专家模型架构", font_size=36, color=self.text_color).to_edge(UP)
        new_subtitle = Text("稀疏、条件计算与动态路由", font_size=24, color=self.text_color)
        new_subtitle.next_to(new_title, DOWN, buff=0.3)
        
        # 转换动画
        self.play(
            FadeOut(self.traditional_title),
            Write(new_title),
            Write(new_subtitle),
            run_time=1.2  # 加快动画速度
        )
        
        # 添加转换说明 - 使用更醒目的颜色和动画效果
        transform_text = Text("Transforming to Sparse Architecture...", 
                              font_size=28, color=YELLOW)
        transform_text.to_edge(DOWN, buff=1.2)  # 增加底部间距
        self.play(Write(transform_text), run_time=0.8)
        self.play(Flash(transform_text, color=YELLOW, flash_radius=0.8, line_length=0.5))

        # 分步展示MoE组件 - 添加解释文本和视觉强调
        component_descriptions = [
            "门控网络决定激活哪些专家",
            "稀疏连接降低计算成本",
            "专业化的专家网络处理不同的输入模式",
            "专家和输出之间的动态路由",
            "清晰的组件分离提高了模型可解释性",
            "输出层组合专家预测"
        ]
        
        component_descriptions_en = [
            "Gating network decides which experts to activate",
            "Sparse connections reduce computational cost",
            "Specialized expert networks process different input patterns",
            "Dynamic routing between experts and output",
            "Clear component separation improves model interpretability",
            "Output layer combines expert predictions"
        ]
        moe_components = [
            self.moe_group[1],  # 门控网络
            self.moe_group[4],  # 门控连接
            self.moe_group[2],  # 专家层
            self.moe_group[5],  # 专家连接
            self.moe_group[6],  # 标签
            self.moe_group[3]   # 输出层
        ]

        # 添加组件高亮颜色
        highlight_colors = [self.gate_color, YELLOW_A, self.expert_colors[0], self.expert_colors[1], WHITE, self.output_color]
        
        description_text = Text("", font_size=24, color=self.text_color)
        description_text.to_edge(DOWN, buff=0.7)  # 调整位置避免遮挡
        
        # 跟踪已显示的组件，以便在添加新组件前确保正确的z-order
        shown_components = VGroup()
        
        for i, (comp, desc, highlight_color) in enumerate(zip(moe_components, component_descriptions, highlight_colors)):
            # 更新描述文本
            new_desc = Text(desc, font_size=24, color=self.text_color)
            new_desc_en = Text(component_descriptions_en[i], font_size=20, color=self.text_color)
            new_desc_group = VGroup(new_desc, new_desc_en).arrange(DOWN, buff=0.2)
            new_desc_group.to_edge(DOWN, buff=0.7)  # 调整位置避免遮挡
            
            # 添加背景框增强可读性
            desc_bg = BackgroundRectangle(new_desc_group, color=BLACK, fill_opacity=0.3)
            desc_with_bg = VGroup(desc_bg, new_desc_group)
            
            if i == 0:
                self.play(
                    FadeOut(transform_text),
                    FadeIn(comp, shift=RIGHT),
                    FadeIn(desc_with_bg),
                    run_time=1.2  # 加快动画速度
                )
                # 添加高亮效果
                self.play(Flash(comp, color=highlight_color, flash_radius=0.6, line_length=0.4))
            else:
                # 确保新组件在z-order上正确显示
                self.remove(comp)
                self.add(comp)
                
                self.play(
                    FadeIn(comp, shift=RIGHT),
                    FadeOut(description_text),
                    FadeIn(desc_with_bg),
                    run_time=1.2  # 加快动画速度
                )
                # 添加高亮效果
                self.play(Flash(comp, color=highlight_color, flash_radius=0.6, line_length=0.4))
            
            shown_components.add(comp)
            description_text = desc_with_bg
            self.wait(0.5)  # 减少等待时间

        self.moe_title = VGroup(new_title, new_subtitle)
        self.description_text = description_text
        self.wait(0.8)  # 减少等待时间

    def demo_dynamic_routing(self):
        """动态路由演示 - 增强视觉效果和交互性"""
        # 清除之前的描述文本和可能存在的其他元素，确保场景干净
        self.play(
            FadeOut(self.description_text),
            *[FadeOut(obj) for obj in self.mobjects if obj != self.moe_group and obj != self.moe_title]
        )
        
        # 添加路由演示标题
        routing_title = Text("动态专家路由演示", 
                            font_size=32, color=self.text_color)
        routing_title.to_edge(UP, buff=0.2)
        
        # 添加解释字幕 - 放在顶部避免遮挡动画内容
        self.show_subtitle("MoE模型根据输入数据动态选择最合适的专家进行处理", duration=3.0, 
                          position="top", 
                          en_text="MoE models dynamically select the most appropriate experts based on input data")
        
        self.play(
            FadeOut(self.moe_title),
            FadeIn(routing_title)
        )
        
        # 创建输入数据点 - 使用更生动的视觉效果
        input_data = VGroup(
            Circle(radius=0.2, color=self.input_color, fill_opacity=0.8),
            Text("x", font_size=20, color=WHITE)
        )
        input_data[1].move_to(input_data[0])
        # 确保输入数据点精确定位
        input_data.move_to(self.moe_group[0][0].get_center())
        
        # 添加输入数据说明 - 调整位置避免遮挡
        input_desc = Text("输入数据样本", font_size=24, color=self.text_color)
        input_desc.next_to(input_data, LEFT, buff=0.5)
        
        # 添加闪烁效果使输入数据更加醒目
        flash_animation = [Flash(input_data, color=self.input_color, flash_radius=0.5, line_length=0.3)]
        
        self.play(
            FadeIn(input_data),
            FadeIn(input_desc),
            *flash_animation,
            run_time=1.0
        )
        self.wait(0.3)  # 减少等待时间
        
        # 门控计算动画 - 使用更清晰的公式和解释
        gate_eq = MathTex(r"G(x) = \text{Softmax}(W_g \cdot x + b_g)", font_size=32, color=YELLOW)
        gate_desc = Text("门控网络计算路由概率", font_size=20, color=self.text_color)
        gate_eq_box = VGroup(gate_eq, gate_desc).arrange(DOWN, buff=0.2)
        
        # 添加背景框增强公式可读性
        gate_eq_bg = BackgroundRectangle(gate_eq_box, color=BLACK, fill_opacity=0.3)
        gate_eq_with_bg = VGroup(gate_eq_bg, gate_eq_box)
        
        # 添加解释字幕 - 放在顶部避免遮挡动画内容
        self.show_subtitle("门控网络分析输入数据，决定将数据路由到哪些专家", duration=3.0, position="top")
        
        # 调整公式位置，确保不会遮挡关键动画元素
        gate_eq_with_bg.to_edge(DOWN, buff=1.5)
        
        self.play(
            Write(gate_eq_with_bg),
            FadeOut(input_desc),
            run_time=1.2
        )

        # 数据流向门控网络 - 添加轨迹效果并优化路径
        # 创建从输入到门控网络的弧线路径
        start_point = input_data.get_center()
        end_point = self.moe_group[1][0][0].get_center()
        # 计算路径的控制点 - 调整曲线形状避免穿过其他元素
        control_point = np.array([
            (start_point[0] + end_point[0]) / 2,
            (start_point[1] + end_point[1]) / 2 + 0.6,  # 增加弯曲度
            0
        ])
        
        # 创建贝塞尔曲线路径
        path_to_gate = CubicBezier(
            start_point,
            start_point * 0.6 + control_point * 0.4,  # 调整控制点权重
            end_point * 0.6 + control_point * 0.4,
            end_point
        )
        
        # 创建数据点副本用于动画，保留原始数据点
        data_copy = input_data.copy()
        
        # 添加轨迹效果，使数据流动更加可视化
        path_dots = VGroup(*[Dot(color=self.input_color, radius=0.03) for _ in range(15)])
        
        # 添加字幕解释数据流向门控网络 - 放在顶部避免遮挡
        self.show_subtitle("输入数据首先被送入门控网络进行分析", duration=2.0, 
                          position="top",
                          en_text="Input data is first sent to the gating network for analysis")
        
        # 创建数据流动动画，添加轨迹效果
        self.play(
            MoveAlongPath(data_copy, path_to_gate),
            *[MoveAlongPath(dot, path_to_gate, rate_func=lambda t: max(0, min(1, 3*t - 0.2*i))) 
              for i, dot in enumerate(path_dots)],
            run_time=2.0,  # 放慢动画速度
            rate_func=smooth
        )
        # 移除数据点副本，避免重叠
        self.remove(data_copy)
        
        # 门控网络激活效果 - 使用LaggedStart同时激活以减少等待时间
        gate_flashes = []
        gate_fills = []
        for gate in self.moe_group[1]:
            gate_fills.append(gate[0].animate.set_fill(self.gate_color, opacity=0.8))
            gate_flashes.append(Flash(gate[0], color=self.gate_color, line_length=0.3))
        
        # 添加字幕解释门控网络激活
        self.show_subtitle("门控网络分析输入数据特征，计算每个专家的激活概率", duration=2.5, 
                          en_text="The gating network analyzes input features and calculates activation probabilities for each expert")
        
        self.play(
            LaggedStart(*gate_fills, lag_ratio=0.3),
            LaggedStart(*gate_flashes, lag_ratio=0.3),
            run_time=1.8  # 放慢动画速度
        )
        
        # 生成更有趣的门控概率分布
        gate_probs = np.array([0.7, 0.2, 0.1])  # 使用固定值以便更好地展示稀疏激活
        
        # 显示门控概率 - 使用更生动的视觉效果
        gate_outputs = VGroup()
        
        # 创建概率条形图，使专家选择过程更加可视化
        prob_bars = VGroup()
        max_bar_height = 1.5
        bar_width = 0.2
        
        for i, prob in enumerate(gate_probs):
            # 创建概率值显示
            prob_text = Text(f"{prob:.1f}", font_size=16, color=WHITE)
            
            # 创建概率条形图
            bar_height = prob * max_bar_height
            bar = Rectangle(
                height=bar_height, 
                width=bar_width, 
                fill_color=self.expert_colors[i], 
                fill_opacity=0.7,
                stroke_color=WHITE,
                stroke_width=1
            )
            
            # 将概率条放在门控神经元旁边
            bar.next_to(self.moe_group[1][i], RIGHT, buff=0.3)
            bar.align_to(self.moe_group[1][i], DOWN)
            
            # 将概率值放在条形图上方
            prob_text.next_to(bar, UP, buff=0.1)
            
            # 组合概率值和条形图
            prob_group = VGroup(bar, prob_text)
            gate_outputs.add(prob_group)
            prob_bars.add(bar)
        
        # 添加字幕解释概率分布
        self.show_subtitle("门控网络输出概率分布，决定每个专家的激活程度", duration=2.5,
                          position="top",
                          en_text="The gating network outputs a probability distribution, determining the activation level of each expert")
        
        # 显示门控输出，使用生长动画使条形图更加生动
        bar_anims = [GrowFromEdge(bar, DOWN) for bar in prob_bars]
        prob_anims = [Write(gate_outputs[i][1]) for i in range(len(gate_outputs))]
        
        self.play(
            LaggedStart(*bar_anims, lag_ratio=0.3),
            LaggedStart(*prob_anims, lag_ratio=0.3),
            run_time=1.5
        )
        self.wait(1.0)
        
        # 展示专家激活 - 根据门控概率
        expert_activations = []
        for i, (expert_group, prob) in enumerate(zip(self.moe_group[2], gate_probs)):
            expert = expert_group[0]  # 获取专家神经元组
            
            # 创建从门控到专家的路径 - 优化路径避免穿过其他元素
            start_point = self.moe_group[1][i][0].get_center()
            end_point = expert[0].get_center()
            # 计算路径的控制点 - 调整曲线形状
            control_point = np.array([
                (start_point[0] + end_point[0]) / 2,
                (start_point[1] + end_point[1]) / 2 - 0.4,  # 增加弯曲度
                0
            ])
            # 创建贝塞尔曲线路径
            path_to_expert = CubicBezier(
                start_point,
                start_point * 0.6 + control_point * 0.4,  # 调整控制点权重
                end_point * 0.6 + control_point * 0.4,
                end_point
            )
            
            # 创建数据点副本
            data_copy = input_data.copy()
            data_copy.scale(prob * 1.2 + 0.3)  # 根据概率调整大小
            
            # 添加到激活列表
            expert_activations.append((data_copy, path_to_expert, expert, prob))
        
        # 按概率排序，先显示高概率的专家
        expert_activations.sort(key=lambda x: x[3], reverse=True)
        
        # 创建用于存储激活的数据点的列表，以便后续清理
        active_data_points = []
        
        # 添加字幕解释专家激活过程
        self.show_subtitle("输入数据被路由到不同专家，激活程度由门控网络决定", duration=2.5, 
                          en_text="Input data is routed to different experts, with activation levels determined by the gating network")
        
        # 展示专家激活
        for i, (data, path, expert_neurons, prob) in enumerate(expert_activations):
            # 只有当概率大于阈值时才激活专家
            if prob > 0.05:
                # 添加数据点闪烁效果，增强视觉吸引力
                self.play(
                    FadeIn(data),
                    Flash(data, color=self.expert_colors[i], flash_radius=0.3, line_length=0.2),
                    run_time=0.5
                )
                active_data_points.append(data)  # 添加到列表以便后续清理
                
                # 为每个专家添加特定解释
                expert_explanations = [
                    "主要专家接收最高激活，处理最相关的特征",
                    "次要专家接收部分激活，处理辅助特征",
                    "低激活专家几乎不参与计算，节省资源"
                ]
                
                expert_explanations_en = [
                    "Primary expert receives highest activation, processing the most relevant features",
                    "Secondary expert receives partial activation, processing auxiliary features",
                    "Low-activation expert barely participates in computation, saving resources"
                ]
                
                # 添加背景框增强文本可读性
                if i < len(expert_explanations):
                    self.show_subtitle(expert_explanations[i], duration=1.5, en_text=expert_explanations_en[i])
                
                # 创建轨迹效果，使数据流动更加可视化
                path_dots = VGroup(*[Dot(color=self.expert_colors[i], radius=0.03) for _ in range(10)])
                
                self.play(
                    MoveAlongPath(data, path),
                    *[MoveAlongPath(dot, path, rate_func=lambda t: max(0, min(1, 3*t - 0.2*j))) 
                      for j, dot in enumerate(path_dots)],
                    run_time=1.5 * prob + 0.8  # 根据概率调整动画时间，但总体放慢
                )
                
                # 专家激活效果 - 使用更生动的视觉效果
                expert_fills = [neuron.animate.set_fill(self.expert_colors[i], opacity=min(0.8, prob + 0.3)) 
                               for neuron in expert_neurons]
                expert_flashes = [Flash(neuron, color=self.expert_colors[i], line_length=0.3) 
                                 for neuron in expert_neurons]
                
                self.play(
                    LaggedStart(*expert_fills, lag_ratio=0.1),
                    LaggedStart(*expert_flashes, lag_ratio=0.1),
                    run_time=1.0
                )
                
                # 添加专家处理解释
                if i == 0:  # 主要专家
                    self.show_subtitle("专家根据自身专长处理输入数据的特定方面", duration=2.0,
                                      en_text="Experts process specific aspects of input data according to their specialization")
                
                # 展示专家到输出的数据流动 - 增强视觉效果
                # 只为高激活的专家(概率>0.2)展示到输出的流动
                if prob > 0.2:
                    # 创建从专家到输出的路径
                    expert_to_output_paths = []
                    for j, src_neuron in enumerate(expert_neurons):
                        # 为每个专家神经元只连接到对应的输出神经元，减少交叉
                        out_idx = j % len(self.moe_group[3])
                        dst_neuron = self.moe_group[3][out_idx]
                        
                        # 计算路径的起点和终点
                        start_point = src_neuron.get_boundary_point(dst_neuron.get_center() - src_neuron.get_center())
                        end_point = dst_neuron.get_boundary_point(src_neuron.get_center() - dst_neuron.get_center())
                        
                        # 添加垂直偏移以避免路径重叠
                        vertical_offset = (j - len(expert_neurons)/2) * 0.05
                        
                        # 创建控制点
                        control_point1 = start_point + RIGHT * 0.5 + UP * vertical_offset
                        control_point2 = end_point + LEFT * 0.5 + UP * vertical_offset
                        
                        # 创建贝塞尔曲线路径
                        path = CubicBezier(
                            start_point,
                            control_point1,
                            control_point2,
                            end_point
                        )
                        expert_to_output_paths.append(path)
                    
                    # 创建数据点 - 大小与专家激活程度成比例
                    output_data_points = []
                    for j in range(len(expert_to_output_paths)):
                        # 创建与专家颜色相同的数据点
                        data_point = Dot(color=self.expert_colors[i], radius=0.08 * prob + 0.04)
                        data_point.move_to(expert_to_output_paths[j].get_start())
                        output_data_points.append(data_point)
                    
                    # 添加字幕解释专家输出
                    if i == 0:  # 只为主要专家添加解释
                        self.show_subtitle("专家处理后的结果被发送到输出层进行组合", duration=2.0,
                                          en_text="Results processed by experts are sent to the output layer for combination")
                    
                    # 创建数据流动动画
                    output_animations = []
                    for j, (path, data_point) in enumerate(zip(expert_to_output_paths, output_data_points)):
                        # 添加数据点
                        self.play(FadeIn(data_point), run_time=0.3)
                        
                        # 创建轨迹效果
                        path_dots = VGroup(*[Dot(color=self.expert_colors[i], radius=0.02) for _ in range(8)])
                        
                        # 创建数据流动动画
                        output_animations.append(MoveAlongPath(data_point, path))
                        output_animations.extend([MoveAlongPath(dot, path, rate_func=lambda t: max(0, min(1, 3*t - 0.2*k))) 
                                                 for k, dot in enumerate(path_dots)])
                    
                    # 播放数据流动动画
                    self.play(
                        LaggedStart(*output_animations, lag_ratio=0.1),
                        run_time=1.5 * prob + 0.5  # 根据概率调整动画时间
                    )
                    
                    # 输出层激活效果
                    output_flashes = []
                    for j, path in enumerate(expert_to_output_paths):
                        out_idx = j % len(self.moe_group[3])
                        output_neuron = self.moe_group[3][out_idx]
                        # 根据专家激活程度调整输出神经元的激活效果
                        output_flashes.append(Flash(output_neuron, color=self.output_color, line_length=0.2 * prob + 0.1))
                    
                    self.play(LaggedStart(*output_flashes, lag_ratio=0.2), run_time=0.8)
        
        self.wait(0.5)
        
        # 清理路由演示 - 确保所有临时元素都被移除
        self.play(
            FadeOut(gate_eq_box),
            FadeOut(gate_outputs),
            FadeOut(input_data),
            *[FadeOut(data) for data in active_data_points],  # 清理所有激活的数据点
            run_time=0.8
        )
        
        self.routing_title = routing_title

    def show_conclusion(self):
        """展示结论和对比"""
        # 清除之前的内容和可能存在的其他元素，确保场景干净
        self.play(
            FadeOut(self.routing_title),
            *[FadeOut(obj) for obj in self.mobjects if obj != self.moe_group]
        )
        
        # 添加结论标题
        conclusion_title = Text("MoE与传统神经网络对比", 
                               font_size=36, color=self.text_color).to_edge(UP)
        
        # 添加总结字幕
        self.show_subtitle("让我们总结MoE模型相比传统神经网络的优势", duration=2.5,
                          en_text="Let's summarize the advantages of MoE models compared to traditional neural networks")
        
        self.play(Write(conclusion_title), run_time=1.0)
        
        # 创建对比表格
        comparison_table = self.create_comparison_table()
        # 调整表格大小和位置，避免遮挡
        comparison_table.scale(0.85).to_edge(LEFT, buff=0.6)
        
        # 添加公式对比 - 调整位置避免重叠
        formula_title = Text("数学公式表示", font_size=28, color=self.text_color)
        formula_title.next_to(comparison_table, RIGHT, buff=1.8).to_edge(UP, buff=1.7)
        
        # 使用已经在setup_parameters中创建的公式组
        self.moe_equations.next_to(formula_title, DOWN, buff=0.6)
        
        # 添加公式标签 - 确保位置精确对齐
        formula_labels = VGroup(
            Text("传统神经网络:", font_size=20, color=BLUE_D).next_to(self.moe_equations[0], LEFT, buff=0.3),
            Text("MoE模型:", font_size=20, color=YELLOW_D).next_to(self.moe_equations[1], LEFT, buff=0.3)
        )
        
        # 展示表格和公式 - 分步显示以避免视觉混乱
        self.play(
            FadeIn(comparison_table),
            run_time=1.2
        )
        
        self.play(
            Write(formula_title),
            run_time=0.8
        )
        
        self.play(
            FadeIn(self.moe_equations),
            Write(formula_labels),
            run_time=1.2
        )
        
        # 添加最终结论
        final_points = [
            "MoE模型通过条件计算实现更好的性能",
            "稀疏激活显著降低计算成本",
            "专家专业化提高模型可解释性",
            "动态路由使模型能够高效处理复杂任务"
        ]
        
        final_points_en = [
            "MoE models achieve better performance through conditional computation",
            "Sparse activation significantly reduces computational cost",
            "Expert specialization improves model interpretability",
            "Dynamic routing enables efficient handling of complex tasks"
        ]
        
        # 添加解释字幕
        self.show_subtitle("MoE模型的核心优势在于条件计算和专家专业化", duration=3.0,
                          en_text="The core advantages of MoE models are conditional computation and expert specialization")
        
        # 创建结论框，优化布局
        conclusion_box = VGroup()
        for i, point in enumerate(final_points):
            bullet = Text("•", font_size=28, color=YELLOW_D)
            text = Text(point, font_size=22, color=self.text_color)
            text_en = Text(final_points_en[i], font_size=18, color=self.text_color.lighter(0.3))
            text_group = VGroup(text, text_en).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
            row = VGroup(bullet, text_group).arrange(RIGHT, buff=0.4)  # 增加间距
            conclusion_box.add(row)
        
        # 调整结论框布局，确保不会与表格重叠
        conclusion_box.arrange(DOWN, aligned_edge=LEFT, buff=0.6)  # 增加行间距
        conclusion_box.next_to(comparison_table, DOWN, buff=1.0).to_edge(LEFT, buff=1.2)  # 增加与表格的间距
        
        # 使用动画效果逐条显示结论点
        conclusion_anims = []
        for i, point in enumerate(conclusion_box):
            conclusion_anims.append(FadeIn(point, shift=UP * 0.3))
        
        self.play(LaggedStart(*conclusion_anims, lag_ratio=0.3), run_time=1.8)
        
        # 最终等待
        self.wait(2)
    
    def create_comparison_table(self):
        """创建对比表格"""
        # 表头
        headers = ["特性", "传统神经网络", "MoE架构"]
        
        # 表格内容
        rows = [
            ["计算方式", "密集 (所有神经元)", "稀疏 (选定的专家)"],
            ["效率", "复杂任务效率较低", "条件计算提高效率"],
            ["专业化", "通用神经元", "任务特定专家"],
            ["扩展性", "性能趋于平稳", "增加专家提升性能"],
            ["可解释性", "黑盒行为", "专家角色更清晰"]
        ]
        
        # 创建表格
        table = VGroup()
        
        # 添加表头
        header_row = VGroup()
        for header in headers:
            cell = Text(header, font_size=22, color=YELLOW)
            header_row.add(cell)
        
        header_row.arrange(RIGHT, buff=0.8, aligned_edge=DOWN)
        table.add(header_row)
        
        # 添加内容行
        for row_data in rows:
            row = VGroup()
            for i, cell_text in enumerate(row_data):
                if i == 0:  # 特性列
                    cell = Text(cell_text, font_size=20, color=WHITE)
                elif i == 1:  # 传统网络列
                    cell = Text(cell_text, font_size=20, color=BLUE_D)
                else:  # MoE列
                    cell = Text(cell_text, font_size=20, color=self.gate_color)
                row.add(cell)
            
            row.arrange(RIGHT, buff=0.8, aligned_edge=DOWN)
            table.add(row)
        
        # 垂直排列所有行
        table.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        
        # 添加背景和边框
        background = BackgroundRectangle(table, color=BLACK, fill_opacity=0.1)
        
        return VGroup(background, table)


if __name__ == "__main__":
    config.frame_size = (1280, 720)
    config.format = "mp4"
    scene = ImprovedMoEComparison()
    scene.render()
