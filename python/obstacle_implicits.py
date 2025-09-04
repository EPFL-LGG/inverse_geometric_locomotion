import torch
from utils import rotate_about_axis

class ImplicitFunction:
    
    def __init__(self, params):
        self.params = params
        self.n_params = params.shape[0]
        
    def check_params(self, new_params):
        '''
        Args:
            new_params (torch tensor of shape (n_params,)): the parameters of the implicit to check
        '''
        assert new_params.shape[0] == self.n_params, "Wrong number of parameters: {} (=new_params.shape[0]) instead of {} (=n_params)".format(new_params.shape[0], self.n_params)
        
    def update_parameters(self, new_params):
        '''
        Args:
            new_params (torch tensor of shape (n_params,)): the parameters of the implicit
            
        Returns:
            func (torch tensor of shape (n_pos,)): the evaluation of the implicit function
        '''
        self.check_params(new_params)
        self.params = new_params

    def evaluate_implicit_function(self, pos):
        '''
        Args:
            pos (torch tensor of shape (n_pos, 3)): where to evaluate the implicit function
            
        Returns:
            func (torch tensor of shape (n_pos,)): the evaluation of the implicit function
        '''
        raise NotImplementedError("Please specify the implicit function to use")
    
    def serialize(self):
        ''' 
        Returns:
            dictionary with parameters and hyperparameters defining the implict
        '''
        return {
            'name': 'ImplicitFunction',
            'params': self.params.detach().tolist()
        }
    
class SphereSquareImplicit(ImplicitFunction):
    
    def __init__(self, params):
        '''
        Args:
            params (torch.tensor of shape (4,)): the sphere center and sphere radius
        '''
        assert params.shape[0] == 4, "Wrong number of parameters, we should have 4 of them (sphere center and radius)"
        super().__init__(params)
        self.sphere_center = None
        self.sphere_radius = None
        self.update_parameters(self.params)
        
    def update_parameters(self, new_params):
        self.check_params(new_params)
        self.sphere_center = new_params[:3]
        self.sphere_radius = new_params[3]
    
    def evaluate_implicit_function(self, pos):
        return torch.sum((pos - self.sphere_center.unsqueeze(0)) ** 2, dim=1) - self.sphere_radius ** 2
    
    def serialize(self):
        return {
            'name': 'SphereSquareImplicit',
            'params': self.params.detach().tolist(),
            'sphere_center': self.sphere_center.detach().tolist(),
            'sphere_radius': self.sphere_radius.detach().tolist(),
        }
    
class SphereImplicit(ImplicitFunction):
    
    def __init__(self, params):
        '''
        Args:
            params (torch.tensor of shape (4,)): the sphere center and sphere radius
        '''
        assert params.shape[0] == 4, "Wrong number of parameters, we should have 4 of them (sphere center and radius)"
        super().__init__(params)
        self.sphere_center = None
        self.sphere_radius = None
        self.update_parameters(self.params)
        
    def update_parameters(self, new_params):
        self.check_params(new_params)
        self.sphere_center = new_params[:3]
        self.sphere_radius = new_params[3]
    
    def evaluate_implicit_function(self, pos):
        return torch.linalg.norm(pos - self.sphere_center.unsqueeze(0), dim=1) - self.sphere_radius
    
    def serialize(self):
        return {
            'name': 'SphereImplicit',
            'params': self.params.detach().tolist(),
            'sphere_center': self.sphere_center.detach().tolist(),
            'sphere_radius': self.sphere_radius.detach().tolist(),
        }
    
class BoxImplicit(ImplicitFunction):
    
    def __init__(self, params):
        '''
        Args:
            params (torch.tensor of shape (6,)): the box center and box xyz-dimensions
        '''
        assert params.shape[0] == 6, "Wrong number of parameters, we should have 6 of them (box center and box dimensions)"
        super().__init__(params)
        self.box_center = None
        self.box_dims = None
        self.update_parameters(self.params)
        
    def update_parameters(self, new_params):
        self.check_params(new_params)
        self.box_center = new_params[:3]
        self.box_dims = new_params[3:]
    
    def evaluate_implicit_function(self, pos):
        pos_centered = pos - self.box_center.unsqueeze(0)
        q = torch.abs(pos_centered) - self.box_dims.unsqueeze(0)
        
        outside = torch.linalg.norm(torch.maximum(q, torch.zeros_like(q)), dim=1)
        inside = torch.minimum(torch.max(q, dim=1).values, torch.zeros(size=(q.shape[0],)))
        return outside + inside
    
    def serialize(self):
        return {
            'name': 'BoxImplicit',
            'params': self.params.detach().tolist(),
            'box_center': self.box_center.detach().tolist(),
            'box_dims': self.box_dims.detach().tolist(),
        }
        
class MoonImplicit(ImplicitFunction):
    
    def __init__(self, params):
        '''
        Args:
            params (torch.tensor of shape (4,)): the sphere center and sphere radius
        '''
        assert params.shape[0] == 3, "Wrong number of parameters, we should have 4 of them (sphere center and radius)"
        super().__init__(params)
        self.outer_sphere_radius = None
        self.inner_sphere_radius = None
        self.inner_center_x_offset = None
        self.a = None
        self.b = None
        self.update_parameters(self.params)
        
    def update_parameters(self, new_params):
        self.check_params(new_params)
        self.outer_sphere_radius = new_params[0]
        self.inner_sphere_radius = new_params[1]
        self.inner_center_x_offset = new_params[2]
        self.a = (self.outer_sphere_radius ** 2 - self.inner_sphere_radius ** 2 + self.inner_center_x_offset ** 2) / (2.0 * self.inner_center_x_offset)
        self.b = torch.sqrt(torch.relu(self.outer_sphere_radius ** 2 - self.a ** 2))
    
    def evaluate_implicit_function(self, pos):
        abs_pos_y = torch.abs(pos[:, 1])
        cross = pos[:, 0] * self.b - abs_pos_y * self.a
        mask_correct_max = torch.le((self.inner_center_x_offset * cross).detach(), (self.inner_center_x_offset ** 2 * torch.relu(self.b - abs_pos_y).detach()))
        
        max_sdfs = torch.max(
            - (torch.linalg.norm(pos - torch.tensor([self.inner_center_x_offset, 0.0, 0.0], device=pos.device), dim=1) - self.inner_sphere_radius),
            torch.linalg.norm(pos, dim=1) - self.outer_sphere_radius,
        )
        
        corner_sdfs = torch.sqrt(
            (pos[:, 0] - self.a) ** 2 + (abs_pos_y - self.b) ** 2
        )
        
        return torch.where(mask_correct_max, max_sdfs, corner_sdfs)
    
    
    
    def serialize(self):
        return {
            'name': 'MoonImplicit',
            'params': self.params.detach().tolist(),
            'outer_sphere_radius': self.outer_sphere_radius.detach().tolist(),
            'inner_sphere_radius': self.inner_sphere_radius.detach().tolist(),
            'inner_center_x_offset': self.inner_center_x_offset.detach().tolist(),
        }
    
class EditImplicit(ImplicitFunction):
    def __init__(self, implicit):
        '''
        Args:
            implicit (object of class ImplicitFunction): the implicit to edit
        '''
        self.implicit = implicit
        super().__init__(implicit.params)

    def update_parameters(self, new_params):
        self.implicit.update_parameters(new_params)
        
    def serialize(self):
        return {
            'name': 'EditImplicit',
            'params': self.params.detach().tolist(),
            'base_implicit': self.implicit.serialize(),
        }
        
class ComplementaryImplicit(EditImplicit):
    def __init__(self, implicit):
        super().__init__(implicit)
        
    def evaluate_implicit_function(self, pos):
        return - self.implicit.evaluate_implicit_function(pos)
    
    def serialize(self):
        ser = super().serialize()
        ser['name'] = 'ComplementaryImplicit'
        return ser
    
class OffsetImplicit(EditImplicit):
    def __init__(self, implicit, offset):
        '''
        Args:
            offset (torch tensor representing a scalar): the offset to apply to the implicit function. Positive means shrinking the object.
        '''
        super().__init__(implicit)
        self.offset = offset
        self.n_params = implicit.n_params + 1
        self.update_parameters(torch.cat([implicit.params, offset.reshape(-1,)]))
        
    def evaluate_implicit_function(self, pos):
        return self.implicit.evaluate_implicit_function(pos) + self.offset
    
    def update_parameters(self, new_params):
        self.implicit.update_parameters(new_params[:-1])
        self.offset = new_params[-1]
    
    def serialize(self):
        ser = super().serialize()
        ser['name'] = 'OffsetImplicit'
        ser['offset'] = self.offset.detach().item()
        return ser
    
class RotateImplicit(EditImplicit):
    def __init__(self, implicit, rotation_vector):
        '''
        Args:
            rotation_vector (torch tensor of shape (3,)): the rotation vector to apply to the implicit function
        '''
        super().__init__(implicit)
        self.rotation_vector = rotation_vector
        self.n_params = implicit.n_params + 3
        self.update_parameters(torch.cat([implicit.params, rotation_vector]))
        
    def evaluate_implicit_function(self, pos):
        return self.implicit.evaluate_implicit_function(rotate_about_axis(pos, self.rotation_axis, -self.rotation_angle))
    
    def update_parameters(self, new_params):
        self.params = new_params
        self.implicit.update_parameters(new_params[:-3])
        self.rotation_vector = new_params[-3:]
        self.rotation_angle = torch.linalg.norm(self.rotation_vector)
        self.rotation_axis = self.rotation_vector / (self.rotation_angle + 1.0e-6)
    
    def serialize(self):
        ser = super().serialize()
        ser['name'] = 'RotateImplicit'
        ser['rotation_vector'] = self.rotation_vector.detach().tolist()
        return ser
    
class TranslateImplicit(EditImplicit):
    def __init__(self, implicit, translation_vector):
        '''
        Args:
            translation_vector (torch tensor of shape (3,)): the translation vector to apply to the implicit function
        '''
        super().__init__(implicit)
        self.translation_vector = translation_vector
        self.n_params = implicit.n_params + 3
        self.update_parameters(torch.cat([implicit.params, translation_vector]))
        
    def evaluate_implicit_function(self, pos):
        return self.implicit.evaluate_implicit_function(pos - self.translation_vector.unsqueeze(0))
    
    def update_parameters(self, new_params):
        self.params = new_params
        self.implicit.update_parameters(new_params[:-3])
        self.translation_vector = new_params[-3:]
    
    def serialize(self):
        ser = super().serialize()
        ser['name'] = 'RotateImplicit'
        ser['translation_vector'] = self.translation_vector.detach().tolist()
        return ser
    
class ScaleImplicit(EditImplicit):
    def __init__(self, implicit, scales):
        '''
        Args:
            scales (torch tensor of shape (3,)): the scaling factor in each direction
        '''
        super().__init__(implicit)
        self.scales = scales
        self.update_parameters(torch.cat([implicit.params, scales]))
        self.n_params = implicit.n_params + 3
        
    def evaluate_implicit_function(self, pos):
        '''The mean is used to preserve the SDF property in case of uniform scaling'''
        return self.implicit.evaluate_implicit_function(pos / self.scales.unsqueeze(0)) * torch.mean(self.scales)
    
    def update_parameters(self, new_params):
        self.params = new_params
        self.implicit.update_parameters(new_params[:-3])
        self.scales = new_params[-3:]
    
    def serialize(self):
        ser = super().serialize()
        ser['name'] = 'ScaleImplicit'
        ser['scales'] = self.scales.detach().tolist()
        return ser
    
class CombineImplicits(ImplicitFunction):
    
    def __init__(self, list_implicits):
        '''
        Args:
            list_implicits (list of objects of class ImplicitFunction): the implicits to combine
        '''
        self.list_implicits = list_implicits
        self.n_implicits = len(list_implicits)
        params = torch.cat([imp.params for imp in list_implicits], dim=0)
        super().__init__(params)
        self.n_params_per_implicit = torch.tensor([imp.n_params for imp in list_implicits], dtype=torch.int32)
        self.slices_per_implicit = torch.cumsum(torch.cat([
            torch.zeros(size=(1,), dtype=torch.int32),
            self.n_params_per_implicit,
        ], dim=0), dim=0)
        self.slices_per_implicit_lb = self.slices_per_implicit[:-1]
        self.slices_per_implicit_ub = self.slices_per_implicit[1:]
        
    def update_parameters(self, new_params):
        self.check_params(new_params)
        for imp, slb, sub in zip(self.list_implicits, self.slices_per_implicit_lb, self.slices_per_implicit_ub):
            imp.update_parameters(new_params[slb:sub])

    def serialize(self):
        return {
            'name': 'CombineImplicits',
            'params': self.params.detach().tolist(),
            'base_implicits': [imp.serialize() for imp in self.list_implicits],
        }

class UnionImplicit(CombineImplicits):
    
    def __init__(self, list_implicits):
        '''
        Args:
            list_implicits (list of objects of class ImplicitFunction): the implicits to combine
        '''
        super().__init__(list_implicits)
        
    def evaluate_implicit_function(self, pos):
        all_implicits = torch.stack([imp.evaluate_implicit_function(pos) for imp in self.list_implicits], dim=1)
        return torch.min(all_implicits, dim=1).values
    
    def serialize(self):
        ser = super().serialize()
        ser['name'] = 'UnionImplicit'
        return ser
    
class IntersectionImplicit(CombineImplicits):
    
    def __init__(self, list_implicits):
        '''
        Args:
            list_implicits (list of objects of class ImplicitFunction): the implicits to combine
        '''
        super().__init__(list_implicits)
        
    def evaluate_implicit_function(self, pos):
        all_implicits = torch.stack([imp.evaluate_implicit_function(pos) for imp in self.list_implicits], dim=1)
        return torch.max(all_implicits, dim=1).values
    
    def serialize(self):
        ser = super().serialize()
        ser['name'] = 'IntersectionImplicit'
        return ser
    
    
############################################
# FAMILIES OF IMPLICITS
############################################

def generate_siggraph_implicit(angle_rotation=torch.pi/5.0, translation=torch.tensor([0.0, 0.0, 0.0]), scale=torch.tensor([1.0, 1.0, 1.0])):
    '''Generates a family of implicits for the SIGGRAPH 2022 paper'''
    sphere_sig1 = SphereImplicit(torch.tensor([0.0, 0.0, 0.0, 1.0]))
    sphere_sig2 = SphereImplicit(torch.tensor([0.0, 2.0, 0.0, 2.2]))
    intersection_implicit_sig = IntersectionImplicit([sphere_sig1, sphere_sig2])
    intersection_scaled_implicit_sig = ScaleImplicit(intersection_implicit_sig, scale)
    rotate_implicit1 = RotateImplicit(intersection_scaled_implicit_sig, torch.tensor([0.0, 0.0, angle_rotation]))
    rotate_implicit2 = RotateImplicit(intersection_scaled_implicit_sig, torch.tensor([0.0, 0.0, angle_rotation + torch.pi]))
    translate_implicit1 = TranslateImplicit(rotate_implicit1, translation)
    translate_implicit2 = TranslateImplicit(rotate_implicit2, - translation + torch.tensor([0.0, 0.0, 0.0]) )
    union_siggraph = UnionImplicit([translate_implicit1, translate_implicit2])
    
    return union_siggraph

